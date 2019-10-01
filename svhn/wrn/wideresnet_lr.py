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
import wideresnet_super as wrnsuper

import super_class


class WideResNet(wrnsuper.WideResNetSuper, super_class.DeepLRModel):
    rank_accumulator = 0
    def __init__(self, depth, num_classes, ranks, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__(
            depth=depth, 
            num_classes=num_classes, 
            ConvF=super_class.Conv2dlr, 
            LinearF=super_class.Linearlr, 
            ranks=ranks, 
            widen_factor=widen_factor, 
            dropRate=dropRate
        )


if __name__ == "__main__":
    import wideresnet_dense
    net0 = wideresnet_dense.WideResNet(depth=16, num_classes=10, ranks=None, widen_factor=8, dropRate=0.4)
    ranks_up = net0.get_ranks()
    print(ranks_up)
    net = WideResNet(depth=16, num_classes=10, ranks=ranks_up, widen_factor=8, dropRate=0.4)
    y = net(torch.randn(1,3,32,32))
    print(y.size())