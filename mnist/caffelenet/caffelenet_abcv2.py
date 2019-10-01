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
import caffelenet_super
import caffelenet_dense

class CaffeLeNet(caffelenet_super.CaffeLeNetSuper, super_class.DeepSP_v2Model):
    def __init__(self, ranks):
        super(CaffeLeNet, self).__init__(ConvF=super_class.Conv2dsp_v2, LinF=super_class.Linearsp_v2, ranks=ranks)

def test():
    net0 = caffelenet_dense.CaffeLeNet()
    ranks_up = net0.get_ranks()
    net = CaffeLeNet(ranks=ranks_up)
    y = net(torch.randn(1, 1, 28, 28))
    print(y.size())

if __name__ == "__main__":
    test()