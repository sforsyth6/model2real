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

def main():
    traindir = "/data/tempor"

    train_dataset = ImageFolderInstance(
            traindir,
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
            train_dataset, batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True)


    model = fcn_resnet50(num_classes=1)
    model = torch.nn.DataParallel(model).cuda()

    criterion = diceLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 0.2,
                                    momentum=0.25,
                                    weight_decay=1e-4)
#    cudnn.benchmark = True


    for epoch in range(200):
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

        out = model(input)["out"]
        loss = criterion(out, target[:,0,:,:].unsqueeze(1).cuda()) 

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = 0.02
    if epoch < 120:
        lr = 0.02
    elif epoch >= 120 and epoch < 160:
        lr = 0.02 * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()