import os
import glob
import numpy as np
import SimpleITK as sitk
import torch
from skimage import morphology
from scipy.interpolate import splprep, splev


def fit_skeleton(mask, label, length):
    # print(label)
    # 提取当前类别的二值mask
    binary_mask = (mask == label)
    
    # 骨架化
    skeleton = morphology.skeletonize(binary_mask)
    
    # 提取骨架点
    coords = np.column_stack(np.nonzero(skeleton))
    
    # 使用样条拟合骨架点
    if len(coords) > 3:
        k = 3

    # Perform spline interpolation using parametric representation
    tck, u = splprep([coords[:, 0], coords[:, 1], coords[:, 2]], s=0, k=k)
    
    # Generate num_points evenly spaced values of the parameter u
    u_new = np.linspace(0, 1, length)
    
    # Evaluate the spline for these new parameter values
    x_new, y_new, z_new = splev(u_new, tck)
    
    # Combine the x, y, z coordinates into a single array
    interp_points = np.vstack((x_new, y_new, z_new)).T
    
    return interp_points


def normalize_coords(coords, size):
    D, H, W = size

    # 将坐标标准化到区间 [-1, 1]
    coords[..., 0] = (coords[..., 0] / (D - 1)) * 2 - 1
    coords[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1
    coords[..., 2] = (coords[..., 2] / (W - 1)) * 2 - 1

    return coords


def process_mask(mask):
    length_dict = {1: 20, 2: 20, 3: 23, 4: 26, 5: 33, 6: 46, 7: 60, 8: 46, 9: 35, 10: 25}
    results = []
    for label in range(1, 21):
        if label == 0:
            continue  # Skip backgrounds
        
        if label > 10:
            length = length_dict[label - 10]
        else:
            length = length_dict[label]

        coords = fit_skeleton(mask, label, length)
        coords = normalize_coords(coords, mask.shape)
        results.append(torch.tensor(coords, dtype=torch.float))
    return results


if __name__ == '__main__':
    path = '../NII/Train_Folder/labelcol/T120-24M.nii.gz'
    roi_path = '../NII/Train_Folder/roi_coords.pth'   

    nii = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(nii)
    roi = torch.load(roi_path)
    roi1 = roi[os.path.basename(path)]
    mask = mask[roi1[0]:roi1[0]+roi1[3], roi1[1]:roi1[1]+roi1[4], roi1[2]:roi1[2]+roi1[5]]

    points = process_mask(mask)

    torch.save(points, './coord_init.pth')
    print(points)
    for i in range(20):
        print(points[i].shape)
        