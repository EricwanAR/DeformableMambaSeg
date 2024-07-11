# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
from utils.tools import AIO_aug_and_post_proc, GetCenterLinePoints


class ImageToImage3D(Dataset):

    def __init__(self, dataset_path: str, augment=False, image_size=(128, 128, 128), val = False, aug_device='cpu') -> None:
        input_path = os.path.join(dataset_path, 'img')
        output_path = os.path.join(dataset_path, 'labelcol')
        path_list = [(os.path.join(input_path, i), os.path.join(output_path, i)) for i in os.listdir(output_path)]
        self.roi = torch.load(os.path.join(dataset_path, 'roi_coords.pth'))
        self.image_size = image_size
        self.val = val
        self.getcoords = GetCenterLinePoints
        if augment:
            self.augment = AIO_aug_and_post_proc(final_size=image_size, rand_rotate=18, do_rand_crop=True, 
                                                 noise=(0, 2), window_norm=(0, 350), 
                                                 rand_gamma=(0.90, 1.11), aug_device=aug_device)
        else:
            self.augment = AIO_aug_and_post_proc(final_size=image_size, window_norm=(0, 350), aug_device=aug_device)
        
        with ThreadPoolExecutor() as executor:
            self.set = list(tqdm(executor.map(lambda p: self.read_data(*p), path_list), total=len(path_list)))

    def read_data(self, img_pth, msk_pth): 
        d = {}
        crop = self.roi[os.path.basename(img_pth)]
        factor = [i.item() / j for i, j in zip(crop[3:], self.image_size)]

        img_full = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(img_pth)), dtype=torch.int16)
        img = img_full[crop[0]:crop[0]+crop[3], crop[1]:crop[1]+crop[4], crop[2]:crop[2]+crop[5]].unsqueeze(0)
        
        msk_full = sitk.ReadImage(msk_pth)
        size = msk_full.GetSize()
        size = (size[2], size[1], size[0])
        spacing = msk_full.GetSpacing()
        spacing = (spacing[2]*factor[0], spacing[1]*factor[1], spacing[0]*factor[2])
        
        msk_full = torch.tensor(sitk.GetArrayFromImage(msk_full), dtype=torch.uint8)
        msk = msk_full[crop[0]:crop[0]+crop[3], crop[1]:crop[1]+crop[4], crop[2]:crop[2]+crop[5]]

        d['image'] = img
        d['mask'] = msk
        d['ori_size'] = size
        d['spacing'] = spacing
        return d

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        image, label, ori_size, spacing = self.set[idx]['image'], self.set[idx]['mask'], self.set[idx]['ori_size'], self.set[idx]['spacing']
        image, label = self.augment(image, label, ori_size)
        sample = {'image': image, 'label': label}
        if self.val:
            sample['spacing'] = spacing
        else:
            sample['coords'] = self.getcoords(label)
        return sample

