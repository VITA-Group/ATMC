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
import mobilenetv2_super
import mobilenetv2_dense

class MobileNetV2(mobilenetv2_super.MobileNetV2, super_class.DeepLRModel):
    def __init__(self, ranks, alpha=1.0, num_classes=10, conv=super_class.Conv2dlr, linear=super_class.Linearlr):
        super(MobileNetV2, self).__init__(alpha, num_classes, ranks, conv=conv, linear=linear)

def test():
    net = mobilenetv2_dense.MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    print(net.get_ranks())
    net1 = MobileNetV2(1.0, 10, net.get_ranks())
    y = net1(x)
    print(y.size())

if __name__=="__main__":
    test()

# '''ResNet in PyTorch.
# For Pre-activation ResNet, see 'preact_resnet.py'.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys
# import os
# file_path = os.path.dirname(__file__)
# sys.path.append(os.path.join(file_path, '../../super_module'))
# sys.path.append(os.path.join(file_path, '../'))
# sys.path.append(file_path)
# import super_class
# import resnet_super
# import resnet_dense

# class BasicBlock(resnet_super.BasicBlockSuper):
#     def __init__(self, in_planes, planes, ConvF=super_class.Conv2dlr, ranks=None, stride=1):
#         super(BasicBlock, self).__init__(in_planes=in_planes, planes=planes, ConvF=ConvF, ranks=ranks, stride=stride)

# class Bottleneck(resnet_super.BottleneckSuper):
#     def __init__(self, in_planes, planes, ConvF=super_class.Conv2dlr, ranks=None, stride=1):
#         super(Bottleneck, self).__init__(in_planes=in_planes, planes=planes, ConvF=ConvF, ranks=ranks, stride=stride)

# class ResNet(resnet_super.ResNetSuper, super_class.DeepLRModel):
#     def __init__(self, block, num_blocks, ranks, num_classes=10):
#         super(ResNet, self).__init__(block=block, num_blocks=num_blocks, ConvF=super_class.Conv2dlr, LinF=super_class.Linearlr, ranks=ranks, num_classes=num_classes)

# def ResNet18(ranks_up):
#     return ResNet(BasicBlock, [2,2,2,2], ranks=ranks_up)

# def ResNet34(ranks_up):
#     return ResNet(BasicBlock, [3,4,6,3], ranks=ranks_up)

# def ResNet50(ranks_up):
#     return ResNet(Bottleneck, [3,4,6,3], ranks=ranks_up)

# def ResNet101(ranks_up):
#     return ResNet(Bottleneck, [3,4,23,3], ranks=ranks_up)

# def ResNet152(ranks_up):
#     return ResNet(Bottleneck, [3,8,36,3], ranks=ranks_up)


# def test():
#     model0 = resnet_dense.ResNet18()
#     ranks_up = model0.get_ranks()
#     # print(ranks_up)
#     net = ResNet18(ranks_up).cuda()
#     data = torch.randn(1,3,32,32)
#     y = net(data.cuda())
#     print(y)
#     # print(net.modules())

# if __name__ == "__main__":
#     test()
