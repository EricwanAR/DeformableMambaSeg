import argparse
import os
import random
import logging
import numpy as np
import time
import torch
import SimpleITK as sitk
from utils import criterions

local_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='M3VPN-CCSeg', type=str)

parser.add_argument('--time', default=local_time, type=str)

# DataSet Information
parser.add_argument('--root', default='../NII', type=str)

parser.add_argument('--test_dir', default='Test_Folder', type=str)

parser.add_argument('--mode', default='test', type=str)

parser.add_argument('--dataset', default='NII', type=str)

parser.add_argument('--model_name', default='ResUNet', type=str)

parser.add_argument('--image_channels', default=1, type=int)

parser.add_argument('--image_size', default=128, type=int)

parser.add_argument('--num_class', default=21, type=int)

parser.add_argument('--seed', default=1111231, type=int)

parser.add_argument('--no_cuda', default=False, action='store_true')

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--save_freq', default=100, type=int)

parser.add_argument('--load', default='', type=str)

parser.add_argument('--skip_infer', default=False, action='store_true')

args = parser.parse_args()


def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
    log_file = os.path.join(log_dir, 'Test-' + args.experiment + '-' + args.time + '.txt')
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.skip_infer:
        logging.info('Skipped inference, try using existing inference for test.')
    else:
        for group in ['G1', 'G2', 'G3', 'G4']:
            test_folder = os.path.join(os.path.abspath(args.root), args.test_dir, group, 'img')
            output_folder = os.path.join('./test/pred', group)
            logging.info(f'{group} inferencing...')
            os.system(f'python infer.py --load {args.load} --dir {test_folder} --output {output_folder} --image_size {args.image_size}')

    dice_dict = {}
    nsd_dict = {}
    dices = []
    nsds = []
            
    for group in ['G1', 'G2', 'G3', 'G4']:
        logging.info('{} test session'.format(group))
        target_folder = os.path.join(os.path.abspath(args.root), args.test_dir, group, 'labelcol')
        output_folder = os.path.join('./test/pred', group)
        names = [i for i in os.listdir(target_folder) if i.endswith('.nii.gz')]
        val_dice = []
        val_nsd = []
        for name in names:
            pred_file = sitk.ReadImage(os.path.join(output_folder, name))
            pred = one_hot_encoder(torch.tensor(sitk.GetArrayFromImage(pred_file)).to(torch.int8).unsqueeze(0), 21)
            target_file = sitk.ReadImage(os.path.join(target_folder, name))
            target = one_hot_encoder(torch.tensor(sitk.GetArrayFromImage(target_file)).to(torch.int8).unsqueeze(0), 21)
            spacing = target_file.GetSpacing()
            spacing = (spacing[2], spacing[1], spacing[0])

            dice = criterions.show_dice(pred, target).item()
            val_dice.append(dice)
            nsd = criterions.show_nsd(pred, target, spacing).item()
            val_nsd.append(nsd)
            logging.info('{} sample {} testing DICE {:.5f}, NSD {:.5f}'.format(group, name, dice, nsd))

        logging.info('{} testing average DICE {:.5f}, NSD {:.5f}'.format(
            group, sum(val_dice)/len(val_dice), sum(val_nsd)/len(val_nsd)))
        dice_dict[group] = val_dice
        nsd_dict[group] = val_nsd
        dices += val_dice
        nsds += val_nsd

    logging.info(f'\nAll Average testing DICE {sum(dices)/len(dices):.5f}, NSD {sum(nsds)/len(nsds):.5f}')
    torch.save({'dice': dice_dict, 'nsd': nsd_dict}, f'./test/{local_time}.pth')

    logging.info('----------------------------------The testing process finished!-----------------------------------')


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
    return output_tensor


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def val_collate_fn(batch):
    # 'batch' 是一个包含从 `Dataset.__getitem__` 返回的多个结果的列表
    # 包含'img', 'label', 'spacing'等键
    # 使用字典推导式来构造新的批次字典
    # 对于图像和标签，默认使用torch.stack来合并
    # 对于间距等非张量数据，保留为列表或转换为张量
    batched_data = {
        'image': torch.stack([item['image'] for item in batch], dim=0),
        'label': torch.stack([item['label'] for item in batch], dim=0),
        # 对于非张量数据，如间距，我们可以简单地保留为列表
        'spacing': [item['spacing'] for item in batch]
    }

    return batched_data


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    main_worker()
