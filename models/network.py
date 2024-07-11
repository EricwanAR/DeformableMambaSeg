import torch
import torch.nn as nn
from mamba_ssm import Mamba


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class SliceSelection(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(SliceSelection, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.gap = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Linear(channels * 8, 6)

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c4 = self.gap(c4)
        feature = c4.view(c4.shape[0], -1)
        out = self.fc(feature)
        return out


class MambaCoordNet(nn.Module):
    def __init__(self, class_num=20, channels=32, d_state=32, d_conv=4, expand=1):
        super(MambaCoordNet, self).__init__()
        self.mambalist = nn.ModuleList(
            [Mamba(d_model=channels, 
                   d_state=d_state, 
                   d_conv=d_conv, 
                   expand=expand) for _ in range(class_num)]
            )
        self.fuse = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.f2c = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        self.coord_init = nn.ParameterList(torch.load('./coord_init.pth'))
        self.class_num = class_num
        lengths = [i.size(0) for i in self.coord_init]
        self.Ns = [0]
        for i in range(class_num):
            self.Ns.append(sum(lengths[:i + 1])) 

    def feature_sample(self, coords, feature_map):
        B, N, _ = coords.shape
        C = feature_map.size(1)
        # 将坐标调整为 grid_sample 所需的格式
        grid = coords.reshape(B, N, 1, 1, 3)
        # 使用 grid_sample 进行插值
        sampled_features = nn.functional.grid_sample(feature_map, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # 调整形状以匹配预期输出
        sampled_features = sampled_features.view(B, C, N).permute(0, 2, 1)

        return sampled_features

    def forward(self, x: torch.Tensor = None):
        B = x.size(0)
        Ns = self.Ns
        
        init = torch.cat(list(self.coord_init), dim=0).repeat(B, 1, 1)
        feat_select = self.feature_sample(init, x.detach())
        complex = torch.cat((init, feat_select), dim=-1)
        fusion = self.fuse(complex)
        
        coord_pred = torch.empty_like(fusion)
        for i in range(self.class_num):
            fusion_current = fusion[:, Ns[i]:Ns[i + 1]]
            coord = self.mambalist[i](fusion_current)
            coord_flip = self.mambalist[i](fusion_current.flip(dims=[-2])).flip(dims=[-2])
            coord_pred[:, Ns[i]:Ns[i + 1]] = (coord + coord_flip) / 2
        
        coord_pred = self.f2c(coord_pred)

        return [coord_pred[:, Ns[i]:Ns[i + 1]] for i in range(self.class_num)]
    

class MambaCoordAttention(nn.Module):
    def __init__(self, class_num=20, in_ch=32, d_state=8, d_conv=4, expand=1):
        super(MambaCoordAttention, self).__init__()
        self.mambalist = nn.ModuleList(
            [Mamba(d_model=in_ch, 
                   d_state=d_state, 
                   d_conv=d_conv, 
                   expand=expand) for _ in range(class_num)]
            )
        self.class_num = class_num

    def coord_normalized2int(self, coords, size):
        D, H, W = size
        int_coords = torch.empty_like(coords)
        int_coords[..., 0] = (coords[..., 0] + 1) / 2 * (D - 1)
        int_coords[..., 1] = (coords[..., 1] + 1) / 2 * (H - 1)
        int_coords[..., 2] = (coords[..., 2] + 1) / 2 * (W - 1)
        int_coords = torch.round(int_coords).int()
        
        lower_bound = torch.tensor([0, 0, 0], device=coords.device)
        upper_bound = torch.tensor([D-1, H-1, W-1], device=coords.device)
        
        return torch.clamp(int_coords, min=lower_bound, max=upper_bound)
    
    def feature_select(self, coords, x):
        B, C, D, H, W = x.shape
        N= coords.size(1)
        
        batch_indices = torch.arange(B).view(B, 1, 1).expand(-1, N, -1)
        channel_indices = torch.arange(C).view(1, 1, C).expand(B, N, -1)
        d_indices = coords[:, :, 0].view(B, N, 1).expand(-1, -1, C)
        h_indices = coords[:, :, 1].view(B, N, 1).expand(-1, -1, C)
        w_indices = coords[:, :, 2].view(B, N, 1).expand(-1, -1, C)
        
        return x[batch_indices, channel_indices, d_indices, h_indices, w_indices]
    
    def update_attention(self, coords, feat_seq, att):
        B, C, D, H, W = att.shape
        N= coords.size(1)

        output_tensor = torch.zeros_like(att)
        count_tensor = torch.zeros(B, 1, D, H, W, device=att.device)

        d_indices = coords[:, :, 0]
        h_indices = coords[:, :, 1]
        w_indices = coords[:, :, 2]

        for b in range(B):
            for n in range(N):
                d, h, w = d_indices[b, n], h_indices[b, n], w_indices[b, n]
                output_tensor[b, :, d, h, w] += feat_seq[b, n]
                count_tensor[b, :, d, h, w] += 1

        # 将累加的特征值取平均
        count_tensor = count_tensor.expand_as(output_tensor)
        output_tensor = output_tensor / count_tensor
        output_tensor = torch.nan_to_num(output_tensor)  # 处理可能的除以零情况

        return att + output_tensor
    
    def mamba_ops(self, feat_seq, seg_label):
        fs_flip = feat_seq.flip(dims=[-2])
        
        y = self.mambalist[seg_label](feat_seq)
        y_flip = self.mambalist[seg_label](fs_flip).flip(dims=[-2])

        return (y + y_flip) / 2

    def forward(self, x, coord):
        size = x.shape[-3:]
        att = torch.zeros_like(x)

        for i in range(self.class_num):
            coord_i = coord[i]
            coord_i = self.coord_normalized2int(coord_i, size)

            feat_select = self.feature_select(coord_i, x)
            feat_sel_att = self.mamba_ops(feat_select, i)
            att = self.update_attention(coord_i, feat_sel_att, att)
        
        return x + att


class DeformableMambaSeg(nn.Module):

    def __init__(self, in_ch=1, channels=32, blocks=3, out_channels=6):
        super(DeformableMambaSeg, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, out_channels)
        
        self.coordnet = MambaCoordNet(out_channels - 1, channels, channels // 4)
        self.coordatt1 = MambaCoordAttention(out_channels - 1, channels, channels // 4)

    def forward(self, input):
        c1 = self.in_conv(input)
        coords = self.coordnet(c1)
        c2 = self.layer1(self.coordatt1(c1, coords))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
   
        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)
        return c8, coords


