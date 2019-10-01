import torch
import torch.nn as nn
import torch.functional as F
import sys
import os

file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, '../../super_module'))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(file_path)
import super_class
import caffelenet_super

class CaffeLeNet(caffelenet_super.CaffeLeNetSuper, super_class.DeepOriginalModel):
    def __init__(self):
        super(CaffeLeNet, self).__init__(ConvF=nn.Conv2d, LinF=nn.Linear)

def test():
    net = CaffeLeNet()
    y = net(torch.randn(1, 1, 28, 28))
    print(y.size())

if __name__ == "__main__":
    test()