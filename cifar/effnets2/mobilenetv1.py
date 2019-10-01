'''
based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, alpha=1.0, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, int(32*alpha), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*alpha))
        self.layers = self._make_layers(in_planes=32, alpha=alpha)
        self.linear = nn.Linear(int(1024*alpha), num_classes)

    def _make_layers(self, in_planes, alpha):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(int(in_planes*alpha), int(out_planes*alpha), stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def param_modules(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                yield module
    
    def empty_all(self):
        for m in self.param_modules():
            m.weight.data.fill_(0.)

class MobileNet_shallow(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), (256,2), (512,2), (1024,2)]

    def __init__(self, alpha=1.0, num_classes=10):
        super(MobileNet_shallow, self).__init__()
        self.conv1 = nn.Conv2d(3, int(32*alpha), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*alpha))
        self.layers = self._make_layers(in_planes=32, alpha=alpha)
        self.linear = nn.Linear(int(1024*alpha), num_classes)

    def _make_layers(self, in_planes, alpha):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(int(in_planes*alpha), int(out_planes*alpha), stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

if __name__=='__main__':
    test()