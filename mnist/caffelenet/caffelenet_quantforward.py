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
    def __init__(self, bit):
        super(CaffeLeNet, self).__init__(ConvF=Conv2d_QuantForward, LinF=Linear_QuantForward, quant_forward=True, bit=bit)

def test():
    net = CaffeLeNet(bit=4)
    net.cuda()
    y = net(torch.randn(1, 1, 28, 28).cuda())
    print(y.size())

if __name__ == "__main__":
    test()