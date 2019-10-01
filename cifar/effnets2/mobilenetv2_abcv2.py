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

class MobileNetV2(mobilenetv2_super.MobileNetV2, super_class.DeepSP_v2Model):
    def __init__(self, ranks, alpha=1.0, num_classes=10, conv=super_class.Conv2dsp_v2, linear=super_class.Linearsp_v2):
        super(MobileNetV2, self).__init__(alpha, num_classes, ranks, conv=super_class.Conv2dsp_v2, linear=super_class.Linearsp_v2)

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