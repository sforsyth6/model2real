import torch
import torch.nn as nn

import numpy as np
import cv2
import time

import torch.backends.cudnn as cudnn

import torchvision.models as models

from dataset import ImageFolderInstance
import torchvision.transforms as transforms

from models.models import fcn_resnet50
from metrics.metrics import diceLoss

from lib.utils import AverageMeter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR', default="/data/tempor",
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.25, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./', type=str, metavar='PATH',
                    help='path to save checkpoint')
parser.add_argument('--loss', default=0, type=int, metavar='PATH',
                    help='0 for dice, 1 for crossentropy')

args = parser.parse_args()

def main():    
    
    train_dataset = ImageFolderInstance(
            args.data,
            transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)


    model = fcn_resnet50(num_classes=1)
    model = torch.nn.DataParallel(model).cuda()

    if not args.loss:
        criterion = diceLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
#    cudnn.benchmark = True


    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # remember best prec@1 and save checkpoint
        if (epoch%5) == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "res50",
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            })


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not args.loss:
            y = target[:,0,:,:].unsqueeze(1).cuda()
        else:
            y =  target[:,0,:,:].type(torch.long).cuda()

        out = model(input)["out"]
        loss = criterion(out,y) 

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

def save_checkpoint(state, filename=args.save_dir + 'checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr
    if epoch < 120:
        lr = args.lr
    elif epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


if __name__ == "__main__":
    main()