import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from concurrent.futures import ThreadPoolExecutor, as_completed


def fit_skeleton(mask, label, length):
    # print(label)
    # 提取当前类别的二值mask
    binary_mask = (mask == label)

    if np.sum(binary_mask) <= 10:
        return np.full((length, 3), fill_value=np.nan)
    
    endpoints = find_endpoints(binary_mask)
    
    # 骨架化
    skeleton = morphology.skeletonize(binary_mask)
    
    for i in range(endpoints.shape[0]):
        d, h, w = endpoints[i]
        skeleton[d, h, w] = True

    # 提取骨架点
    coords = np.column_stack(np.nonzero(skeleton))
    
    # 使用样条拟合骨架点
    if len(coords) > 3:
        k = 3
    else:
        k = len(coords) - 1

    # Perform spline interpolation using parametric representation
    tck, u = splprep([coords[:, 0], coords[:, 1], coords[:, 2]], s=0, k=k)
    
    # Generate num_points evenly spaced values of the parameter u
    u_new = np.linspace(0, 1, length)
    
    # Evaluate the spline for these new parameter values
    x_new, y_new, z_new = splev(u_new, tck)
    
    # Combine the x, y, z coordinates into a single array
    interp_points = np.vstack((x_new, y_new, z_new)).T
    
    return interp_points


def find_endpoints(binary_image):
    """
    找到三维二值化图像中物体的两端端点和形态学上的中点
    :param binary_image: 三维二值化图像
    :return: 端点和形态学中点的坐标，3x3的numpy数组
    """
    # 提取物体的点云
    points = np.argwhere(binary_image)
    
    # 使用PCA计算主轴
    pca = PCA(n_components=1)
    pca.fit(points)
    
    # 计算在主轴方向上的投影
    points_proj = pca.transform(points)
    
    # 沿主轴方向排序点云
    sorted_indices = np.argsort(points_proj, axis=0).flatten()
    sorted_points = points[sorted_indices]
    
    # 获取端点和形态学上的中点
    endpoint1 = sorted_points[0]
    endpoint2 = sorted_points[-1]
    
    # 构造3x3的numpy数组
    points_array = np.array([endpoint1, endpoint2])
    
    return points_array


def normalize_coords(coords, size):
    D, H, W = size

    # 将坐标标准化到区间 [-1, 1]
    coords[..., 0] = (coords[..., 0] / (D - 1)) * 2 - 1
    coords[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1
    coords[..., 2] = (coords[..., 2] / (W - 1)) * 2 - 1

    return coords


def GetCenterLinePoints(seg):
    mask = seg.cpu().numpy()
    uniq_label = np.unique(mask)
    length_dict = {1: 20, 2: 20, 3: 20, 4: 20, 5: 40, 6: 40, 7: 60, 8: 60, 9: 40, 10: 20}
    
    def process_label(label):
        if label > 10:
            length = length_dict[label - 10]
        else:
            length = length_dict[label]
        
        if label not in uniq_label:
            return torch.full((length, 3), float('nan'))
        
        coords = fit_skeleton(mask, label, length)
        coords = normalize_coords(coords, mask.shape)
        return torch.tensor(coords, dtype=torch.float32)
    
    with ThreadPoolExecutor() as executor:
        seg_coords = list(executor.map(process_label, range(1, 21)))
    
    return seg_coords


def gamma_augment(image, gamma_low=0.90, gamma_hi=1.11):
    gamma = torch.rand(1, device=image.device) * (gamma_hi-gamma_low) + gamma_low
    image = image ** gamma
    return image


def windowing(image, window_low=0, window_high=350):
    window_width = window_high - window_low

    # Clamp intensities
    windowed_array = torch.clamp(image, window_low, window_high)

    # Normalize the windowed intensities to the range [0, 1]
    normalized = (windowed_array - window_low) / window_width
    return normalized
    

def rand_crop(image, label=None, ori_size=(600, 512, 512), fixed=None):
    H, W, D = image.shape[-3:]
    if fixed is None:
        start = (ori_size * np.random.random(3) * 0.02).astype(int)
        end = ((H, W, D) - (ori_size * np.random.random(3) * 0.02)).astype(int)
    else:
        start = (np.asarray(ori_size) * fixed).astype(int)
        end = ((H, W, D) - (np.asarray(ori_size) * fixed)).astype(int)

    image = image[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    if label is None:
        return image
    else:
        label = label[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return image, label   

    
def add_gaussian_noise(image, mean=0, std=1):
    noise = torch.normal(mean=mean, std=std, size=image.shape, device=image.device)
    image += noise
    return image


def random_rotate(image, label, max_angle=18):
    B, C, H, W, D = image.shape
    angles = np.radians(np.random.uniform(-max_angle, max_angle, size=3))
    cos_vals, sin_vals = np.cos(angles), np.sin(angles)

    Rx = torch.tensor([[1, 0, 0, 0], [0, cos_vals[0], sin_vals[0], 0], [0, -sin_vals[0], cos_vals[0], 0], [0, 0, 0, 1]], dtype=torch.float32, device=image.device)
    Ry = torch.tensor([[cos_vals[1], 0, -sin_vals[1], 0], [0, 1, 0, 0], [sin_vals[1], 0, cos_vals[1], 0], [0, 0, 0, 1]], dtype=torch.float32, device=image.device)
    Rz = torch.tensor([[cos_vals[2], sin_vals[2], 0, 0], [-sin_vals[2], cos_vals[2], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=image.device)

    R = Rx @ Ry @ Rz

    R = R[:3, :4].unsqueeze(0).repeat(B, 1, 1)

    grid = F.affine_grid(R, size=image.size(), align_corners=False)
    rotated_image = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
    rotated_label = F.grid_sample(label, grid, mode='nearest', align_corners=False)

    return rotated_image, rotated_label


class AIO_aug_and_post_proc(object):
    def __init__(self, final_size=(128, 128, 128), rand_rotate=None,
                 do_rand_crop=False, noise=None, window_norm=None, rand_gamma=None,
                 aug_device='cpu') -> None:
        self.rot = rand_rotate
        self.do_crop = do_rand_crop
        self.noise = noise
        self.norm = window_norm
        self.gamma = rand_gamma
        self.final_size = final_size
        self.device = aug_device

    def __call__(self, image, label, ori_size) -> torch.Any:
        image = image.unsqueeze(0).float().to(self.device)
        label = label.unsqueeze(0).unsqueeze(0).float().to(self.device)

        if self.rot is not None and np.random.uniform() < 0.9:
            image, label = random_rotate(image, label, self.rot)
        if self.do_crop:
            image, label = rand_crop(image, label, ori_size)
        else:
            image, label = rand_crop(image, label, ori_size, 0.01)

        image = F.interpolate(image, size=self.final_size, mode="trilinear")
        label = F.interpolate(label, size=self.final_size, mode="nearest-exact")

        if self.noise is not None and np.random.uniform() < 0.5:
            image = add_gaussian_noise(image, *self.noise)
        if self.norm is not None:
            image = windowing(image, *self.norm)
        if self.gamma is not None and np.random.uniform() < 0.9:
            image = gamma_augment(image, *self.gamma)

        return image.squeeze(0), label.squeeze()

if __name__ == "__main__":
    image = torch.randint(0, 1000, (1,4,4,4)).float()
    label = torch.randint(0, 21, (4,4,4)).float()
    augment = AIO_aug_and_post_proc(rand_pad=2, rand_rotate=18, rand_crop=2, 
                                                 noise=(0, 2), window_norm=(0, 350), 
                                                 rand_gamma=(0.90, 1.11))
    image, label = augment(image, label)
    print(image, label)