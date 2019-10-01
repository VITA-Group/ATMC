import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, '../../super_module'))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(file_path)
import super_class
import wideresnet_super as wrnsuper


class WideResNet(wrnsuper.WideResNetSuper, super_class.DeepOriginalModel):
    rank_accumulator = 0
    def __init__(self, depth, num_classes, ConvF=nn.Conv2d, LinearF=nn.Linear, ranks=None, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__(
            depth=depth, 
            num_classes=num_classes, 
            ConvF=ConvF, 
            LinearF=LinearF, 
            ranks=ranks, 
            widen_factor=widen_factor, 
            dropRate=dropRate
        )


if __name__ == "__main__":
    net = WideResNet(depth=16, num_classes=10, ranks=None, widen_factor=8, dropRate=0.4)
    y = net(torch.randn(1,3,32,32))
    print(y.size())