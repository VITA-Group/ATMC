'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
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
import resnet_super


class BasicBlock(resnet_super.BasicBlockSuper):
    def __init__(self, in_planes, planes, ConvF=nn.Conv2d, stride=1):
        super(BasicBlock, self).__init__(in_planes=in_planes, planes=planes, ConvF=ConvF, stride=stride)

class Bottleneck(resnet_super.BottleneckSuper):
    def __init__(self, in_planes, planes, ConvF=nn.Conv2d, stride=1):
        super(Bottleneck, self).__init__(in_planes=in_planes, planes=planes, ConvF=ConvF, stride=stride)

class ResNet(resnet_super.ResNetSuper, super_class.DeepOriginalModel):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__(block=block, num_blocks=num_blocks, ConvF=nn.Conv2d, LinF=nn.Linear, num_classes=num_classes)

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet34()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    print(net.get_ranks())

if __name__ == "__main__":
    test()
