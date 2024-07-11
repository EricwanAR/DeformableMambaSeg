import argparse
import os
import random
import logging
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.network import DeformableMambaSeg
from utils import criterions
from Load_Dataset import ImageToImage3D


def parse_tuple(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Must be x,y,z")


local_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='M3VPN-CCSeg', type=str)

parser.add_argument('--time', default=local_time, type=str)

# DataSet Information
parser.add_argument('--root', default='../NII', type=str)

parser.add_argument('--train_dir', default='Train_Folder', type=str)

parser.add_argument('--val_dir', default='Val_Folder', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--dataset', default='NII', type=str)

parser.add_argument('--model_name', default='DeformableMambaSeg', type=str)

parser.add_argument('--image_channels', default=1, type=int)

parser.add_argument('--image_size', default=(128, 128, 128), type=parse_tuple, help='Input x,y,z')

# Training Information
parser.add_argument('--lr', default=0.0120, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='CombinedLoss', type=str)

parser.add_argument('--num_class', default=21, type=int)

parser.add_argument('--seed', default=1111231, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=0, type=int)

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--total_epochs', default=500, type=int)

parser.add_argument('--save_freq', default=100, type=int)

parser.add_argument('--resume', default=False, action='store_true')

parser.add_argument('--load', default='', type=str)

args = parser.parse_args()


def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
    log_file = os.path.join(log_dir, args.experiment + '-' + args.time + '.txt')
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = DeformableMambaSeg(in_ch=1, out_channels=21, channels=32, blocks=3)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = criterions.CombinedLoss(dice_weight=2.0, ce_weight=1.0)
    criterion_coord = criterions.BipartiteEuclideanLoss()

    if args.load == '':
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      args.experiment + '-' + args.time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        logging.info('re-training!!!')
    elif os.path.isfile(args.load):
        logging.info('loading checkpoint {}'.format(args.load))
        checkpoint = torch.load(args.load, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])

        if args.resume:
            args.start_epoch = checkpoint['epoch'] + 1
            checkpoint_dir = os.path.dirname(args.load)
        else:
            checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                          args.experiment + '-' + args.time)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.load, args.start_epoch))
    else:
        os.remove(log_file)
        raise ValueError('Invalid checkpoint path to load')

    if args.start_epoch >= args.total_epochs:
        args.total_epochs = args.start_epoch + 100
        logging.info('Invalid total_epochs argument, automatically extended 100 epochs from start epoch.')

    writer = SummaryWriter(os.path.join(checkpoint_dir, 'tensorboard'))

    train_path = os.path.join(os.path.abspath(args.root), args.train_dir)
    val_path = os.path.join(os.path.abspath(args.root), args.val_dir)

    logging.info('Start loading datasets')
    train_set = ImageToImage3D(dataset_path=train_path, augment=True, 
                               image_size=args.image_size, aug_device='cuda')
    val_set = ImageToImage3D(dataset_path=val_path, val=True, 
                             image_size=args.image_size, aug_device='cuda')
    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for validation = {}'.format(len(val_set)))

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=4, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False, drop_last=False,
                            collate_fn=val_collate_fn)

    start_time = time.time()
    best_val_dice = 0
    best_dice_epoch = 0
    # best_val_nsd = 0
    # best_nsd_epoch = 0

    for epoch in range(args.start_epoch, args.total_epochs):
        start_epoch = time.time()

        logging.info('Epoch {} training session'.format(epoch))
        model.train()
        average_loss = 0
        average_loss_coord = 0
        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.total_epochs, args.lr)

            x, target, coords = data['image'], data['label'], data['coords']

            output, coord_pred = model(x.cuda())

            loss_seg = criterion(output, target.cuda())
            loss_coord = criterion_coord(coord_pred, [i.cuda() for i in coords])
            loss = loss_seg + loss_coord
            
            average_loss += loss.item()
            average_loss_coord += loss_coord.item()

            logging.info('Epoch: {}_Iter:{}  loss: {:.5f}'.format(epoch, i, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

        average_loss /= len(train_loader)
        average_loss_coord /= len(train_loader)
        logging.info('Epoch {} average training loss: {:.5f}'.format(epoch, average_loss))

        logging.info('Epoch {} validation session'.format(epoch))
        model.eval()
        with torch.no_grad():
            val_dice = 0
            # val_nsd = 0
            for i, data in enumerate(val_loader):
                x, target, spacing = data['image'], data['label'], data['spacing']

                output, _ = model(x.cuda())
                output, target = criterions.val_preprocess(output, target.cuda())

                dice = criterions.show_dice(output, target)
                val_dice += dice.item()
                # nsd = criterions.show_nsd(output.cpu(), target.cpu(), spacing)
                # val_nsd += nsd.item()

            val_dice /= len(val_loader)
            # val_nsd /= len(val_loader)

        logging.info('Epoch: {}_Validation Dice: {:.5f}'.format(epoch, val_dice))
        # logging.info('Epoch: {}_Validation NSD: {:.5f}'.format(epoch, val_nsd))

        torch.cuda.empty_cache()

        end_epoch = time.time()

        if val_dice > best_val_dice:
            file_name = os.path.join(checkpoint_dir, 'model_best_dice.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            },
                file_name)
            logging.info('!!! New best validation Dice {:.5f}! Model saved.'.format(val_dice))
            best_val_dice = val_dice
            best_dice_epoch = epoch
        else:
            logging.info('Best validation DICE is still {:.5f} from {} epochs ago.'.format(best_val_dice,
                                                                                           epoch - best_dice_epoch))

        # if val_nsd > best_val_nsd:
        #     file_name = os.path.join(checkpoint_dir, 'model_best_nsd.pth')
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optim_dict': optimizer.state_dict()
        #     },
        #         file_name)
        #     logging.info('!!! New best validation NSD {:.5f}! Model saved.'.format(val_nsd))
        #     best_val_nsd = val_nsd
        #     best_nsd_epoch = epoch
        # else:
        #     logging.info(
        #         'Best validation NSD is still {:.5f} from {} epochs ago.'.format(best_val_nsd, epoch - best_nsd_epoch))

        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) % int(args.total_epochs - 1) == 0 \
                or (epoch + 1) % int(args.total_epochs - 2) == 0 \
                or (epoch + 1) % int(args.total_epochs - 3) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            },
                file_name)

        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Training_loss', average_loss, epoch)
        writer.add_scalar('Training_loss_coord', average_loss_coord, epoch)
        writer.add_scalar('Validation_DICE', val_dice, epoch)
        # writer.add_scalar('Validation_NSD', val_nsd, epoch)

        epoch_time_minute = (end_epoch - start_epoch) / 60
        remaining_time_hour = (args.total_epochs - epoch - 1) * epoch_time_minute / 60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.total_epochs - 1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    },
        final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600

    writer.close()

    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=2.7):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / (max_epoch * 1.6), power), 8)


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
