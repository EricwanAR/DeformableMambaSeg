import argparse
import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from models.network import DeformableMambaSeg


parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='../NII/external', type=str)
parser.add_argument('--output', default='./test/infer', type=str)
parser.add_argument('--image_size', default=128, type=int)
parser.add_argument('--load', type=str)
args = parser.parse_args()


def infer(model, img_dir, output_dir, roi_dict):
    paths = glob.glob(os.path.abspath(os.path.join(img_dir, '*.nii.gz')))
    os.makedirs(output_dir, exist_ok=True)

    for path in tqdm(paths):
        roi = roi_dict[os.path.basename(path)]
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)
        array = torch.tensor(array, dtype=torch.float)

        x = torch.nn.functional.interpolate(array[roi[0]:roi[0]+roi[3], roi[1]:roi[1]+roi[4], roi[2]:roi[2]+roi[5]].unsqueeze(0).unsqueeze(0), 
                                            size=(args.image_size, args.image_size, args.image_size), 
                                            mode="trilinear").cuda()
        x = torch.clamp(x, 0, 350) / 350
        y, _ = model(x)
        y = torch.nn.functional.interpolate(y.cpu(), size=tuple(roi[3:]), mode="trilinear")
        roi_map = torch.argmax(y, dim=1).squeeze().numpy()
        map = np.zeros(array.shape, dtype=np.uint8)
        map[roi[0]:roi[0]+roi[3], roi[1]:roi[1]+roi[4], roi[2]:roi[2]+roi[5]] = roi_map

        map = sitk.GetImageFromArray(map)
        map.SetOrigin(image.GetOrigin())
        map.SetDirection(image.GetDirection())
        map.SetSpacing(image.GetSpacing())
    
        sitk.WriteImage(map, os.path.join(output_dir, os.path.basename(path)))


if __name__ == '__main__':
    model = DeformableMambaSeg(in_ch=1, out_channels=21, channels=32, blocks=3).cuda()
    if not os.path.isfile(args.load):
        raise ValueError('Invalid checkpoint path to load') 
    ckpt = torch.load(args.load, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    roi_dict = torch.load(os.path.join(os.path.dirname(args.dir), 'roi_coords.pth'))
    infer(model, args.dir, args.output, roi_dict)