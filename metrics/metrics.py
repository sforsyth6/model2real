import torch.nn as nn
import torch

class diceLoss(nn.Module):
    def __init__(self):
        super(diceLoss, self).__init__()

    def forward(self, x, targets):
        batch_size = x.size(0)

        intersection = torch.mul(x,targets)

        dice = 0

        for i in range(batch_size):
            X = torch.sum(x[i])
            B = torch.sum(targets[i])
            dice += 2*torch.sum(intersection[i])/(X+B)

        dice = dice/batch_size

        return 1 - dice
