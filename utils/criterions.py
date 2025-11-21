import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from monai.metrics.surface_dice import compute_surface_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from .cldice import soft_cldice


class BipartiteEuclideanLoss(nn.Module):
    def __init__(self):
        super(BipartiteEuclideanLoss, self).__init__()

    def forward(self, pred_coords, target_coords):
        S, B = len(pred_coords), pred_coords[0].size(0)
        loss = 0.0
        valid_counts = 0

        for s in range(S):
            scaling = 20 / pred_coords[s].size(1)
            for b in range(B):
                # assert len(pred_coords[s][b]) == len(target_coords[s][b])
                if torch.isnan(target_coords[s][b]).all():
                    continue
                else:
                    valid_counts += 1
                # 计算欧氏距离矩阵
                dist_matrix = torch.cdist(pred_coords[s][b], target_coords[s][b], p=2)
                
                # 使用匈牙利算法求解最小匹配
                row_ind, col_ind = linear_sum_assignment(dist_matrix.cpu().detach().numpy())
                
                # 累加匹配的欧氏距离
                for i, j in zip(row_ind, col_ind):
                    loss += dist_matrix[i, j] * scaling

        return loss / valid_counts


class GreedyBipartiteEuclideanLoss(nn.Module):
    def __init__(self):
        super(GreedyBipartiteEuclideanLoss, self).__init__()

    def forward(self, pred_coords, target_coords):
        B, S, N, _ = pred_coords.shape
        loss = 0.0
        valid_counts = 0

        for b in range(B):
            for s in range(S):
                if torch.isnan(target_coords[b, s]).all():
                    continue
                else:
                    valid_counts += 1
                
                dist_matrix = torch.cdist(pred_coords[b, s], target_coords[b, s], p=2)
                match_indices = self.greedy_match(dist_matrix)
                
                for i, j in match_indices:
                    loss += dist_matrix[i, j]

        return loss / valid_counts

    def greedy_match(self, dist_matrix):
        N = dist_matrix.size(0)
        matched = []
        
        # Initialize indices
        row_indices = torch.arange(N)
        col_indices = torch.arange(N)
        
        while len(row_indices) > 0:
            min_val, min_idx = torch.min(dist_matrix[row_indices][:, col_indices].view(-1), 0)
            i, j = divmod(min_idx.item(), len(col_indices))
            matched.append((row_indices[i].item(), col_indices[j].item()))
            
            # Remove matched indices
            row_indices = row_indices[row_indices != row_indices[i]]
            col_indices = col_indices[col_indices != col_indices[j]]
        
        return matched


def one_hot_encoder(input_tensor, n_classes):
    # encode integer labeled mask tensor of size [B, H, W, ...] or [B, 1, H, W, ...] to one-hot tensor of size [B, C, H, W, ...]
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    if len(input_tensor.shape) == 5:
        output_tensor = torch.cat(tensor_list, dim=1)
    else:
        output_tensor = torch.stack(tensor_list, dim=1)
    return output_tensor.float()


def one_hot_logits(logits):
    # encode integer labeled mask tensor of size [B, C, H, W, ...] to one-hot tensor of the same shape
    # get the index of max logits of each voxel
    _, predicted = torch.max(logits, dim=1, keepdim=True)

    # create one-hot code
    one_hot = torch.zeros_like(logits, dtype=torch.float32)
    return one_hot.scatter_(1, predicted, 1)


def dice_coeff(input, target, exp=1, smooth=1e-5):
    # input is a torch variable of size [B, C, H, W, ...] representing log probabilities for each class
    # target is a one-hot encoded tensor of size [B, C, H, W, ...]
    # return is a tensor of size [B, C] representing the dice of each class of each sample in the batch
    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    inter = torch.sum(input * target, dim=-1)
    union = torch.sum(input ** exp, dim=-1) + torch.sum(target ** exp, dim=-1)

    return (2. * inter + smooth) / (union + smooth)


def custom_ce_loss(input, target, n_classes):
    log_probs = F.log_softmax(input, dim=1)
    loss = 0

    for c in range(n_classes):
        class_mask = (target == c).float()
        class_log_probs = log_probs[:, c, :, :, :]
        class_loss = -torch.sum(class_log_probs * class_mask)
        num_voxels = torch.sum(class_mask)
        if num_voxels > 0:
            class_loss /= num_voxels
        loss += class_loss

    return loss / n_classes


def val_preprocess(input, target):
    input = one_hot_logits(input)
    target = one_hot_encoder(target, n_classes=input.size(1))
    return input, target


def show_dice(input, target):
    dice = dice_coeff(input, target, exp=1)
    
    return dice.mean(1).mean(0)


def show_nsd(input, target, spacing=None):
    nsd = compute_surface_dice(input, target, 
                               class_thresholds=[1 for _ in range(input.size(1)-1)], 
                               spacing=spacing)

    return nsd.nan_to_num(nan=0.).mean(1).mean(0)
    

class DiceLoss(nn.Module):
    def __init__(self, exp=2):
        super(DiceLoss, self).__init__()
        self.exp = exp

    def forward(self, input, target):
        # input is a torch variable of size [B, C, H, W, ...] representing log probabilities for each class
        # target is a long tensor of size [B, H, W, ...]
        input = F.softmax(input, dim=1)
        target = one_hot_encoder(target, n_classes=input.size(1))
        loss = 1 - dice_coeff(input, target, exp=2)

        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, dice_exp=2):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(exp=dice_exp)

    def forward(self, input, target):

        # Cross-Entropy Loss
        CE_loss = custom_ce_loss(input, target, input.size(1))

        # Dice Loss
        dice_loss = self.dice_loss(input, target)

        # Combined Loss
        combined_loss = CE_loss*self.ce_weight + dice_loss*self.dice_weight

        return combined_loss


class CombinedMoreLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, cldice_weight=0.5, dice_exp=2):
        super(CombinedMoreLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cldice_weight = cldice_weight
        self.dice_loss = DiceLoss(exp=dice_exp)
        self.cldice_loss = soft_cldice() 

    def forward(self, input, target):

        # Cross-Entropy Loss
        CE_loss = custom_ce_loss(input, target, input.size(1))

        # Dice Loss
        dice_loss = self.dice_loss(input, target)

        # CLDice Loss
        cldice_loss = self.cldice_loss(input[:, 1:, :, :, :].sum(dim=1, keepdim=True), (target > 0).unsqueeze(1).float())

        # Combined Loss
        combined_loss = CE_loss*self.ce_weight + dice_loss*self.dice_weight + cldice_loss*self.cldice_weight

        return combined_loss