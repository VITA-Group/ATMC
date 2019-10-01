'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlockSuper(nn.Module):
    expansion = 1
    rank_accumulator = 0

    def __init__(self, in_planes, planes, ConvF, ranks=None, stride=1):
        super(BasicBlockSuper, self).__init__()
        if ConvF is nn.Conv2d:
            self.conv1 = ConvF(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = ConvF(in_planes, planes, rank=ranks[0], kernel_size=3, stride=stride, padding=1, bias=False)
            self.rank_accumulator += 1
        self.bn1 = nn.BatchNorm2d(planes)
        if ConvF is nn.Conv2d:
            self.conv2 = ConvF(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = ConvF(planes, planes, rank=ranks[1], kernel_size=3, stride=1, padding=1, bias=False)
            self.rank_accumulator += 1
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if ConvF is nn.Conv2d:
                self.shortcut = nn.Sequential(
                    ConvF(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    ConvF(in_planes, self.expansion*planes, rank=ranks[2], kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
                self.rank_accumulator += 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckSuper(nn.Module):
    expansion = 4
    rank_accumulator = 0

    def __init__(self, in_planes, planes, ConvF, ranks=None, stride=1):
        super(BottleneckSuper, self).__init__()
        if ConvF is nn.Conv2d:
            self.conv1 = ConvF(in_planes, planes, kernel_size=1, bias=False)
        else:
            self.conv1 = ConvF(in_planes, planes, rank=ranks[self.rank_accumulator], kernel_size=1, bias=False)
            self.rank_accumulator += 1
        self.bn1 = nn.BatchNorm2d(planes)
        if ConvF is nn.Conv2d:
            self.conv2 = ConvF(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = ConvF(planes, planes, rank=ranks[self.rank_accumulator], kernel_size=3, stride=stride, padding=1, bias=False)
            self.rank_accumulator += 1
        self.bn2 = nn.BatchNorm2d(planes)
        if ConvF is nn.Conv2d:
            self.conv3 = ConvF(planes, self.expansion*planes, kernel_size=1, bias=False)
        else:
            self.conv3 = ConvF(planes, self.expansion*planes, rank=ranks[self.rank_accumulator], kernel_size=1, bias=False)
            self.rank_accumulator += 1
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if ConvF is nn.Conv2d:
                self.shortcut = nn.Sequential(
                    ConvF(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    ConvF(in_planes, self.expansion*planes, kernel_size=1, rank=ranks[self.rank_accumulator], stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
                self.rank_accumulator += 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetSuper(nn.Module):
    offset = 0
    def __init__(self, block, num_blocks, ConvF, LinF, ranks=None, num_classes=10):
        super(ResNetSuper, self).__init__()
        self.in_planes = 64
        if ConvF is nn.Conv2d and LinF is nn.Linear:
            self.conv1 = ConvF(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], ConvF, stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], ConvF, stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], ConvF, stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], ConvF, stride=2)
            self.linear = LinF(512*block.expansion, num_classes)
        else:
            self.conv1 = ConvF(3, 64, kernel_size=3, rank=ranks[0], stride=1, padding=1, bias=False)
            self.offset += 1
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], ConvF, stride=1, ranks=ranks[self.offset:])
            self.layer2 = self._make_layer(block, 128, num_blocks[1], ConvF, stride=2, ranks=ranks[self.offset:])
            self.layer3 = self._make_layer(block, 256, num_blocks[2], ConvF, stride=2, ranks=ranks[self.offset:])
            self.layer4 = self._make_layer(block, 512, num_blocks[3], ConvF, stride=2, ranks=ranks[self.offset:])
            self.linear = LinF(512*block.expansion, num_classes, rank=ranks[-1])


    def _make_layer(self, block, planes, num_blocks, ConvF, stride, ranks=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # print("{} {}".format(strides, ranks))
        base = 0
        for stride in strides:
            if ranks:
                b_unit = block(self.in_planes, planes, ConvF, ranks[base:], stride)
                self.offset += b_unit.rank_accumulator
                base += b_unit.rank_accumulator
            else:
                b_unit = block(self.in_planes, planes, ConvF, stride)
            layers.append(b_unit)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNetSuper(BasicBlockSuper, [2,2,2,2],nn.Conv2d,nn.Linear)

def ResNet34():
    return ResNetSuper(BasicBlockSuper, [3,4,6,3],nn.Conv2d,nn.Linear)

def ResNet50():
    return ResNetSuper(BottleneckSuper, [3,4,6,3],nn.Conv2d,nn.Linear)

def ResNet101():
    return ResNetSuper(BottleneckSuper, [3,4,23,3],nn.Conv2d,nn.Linear)

def ResNet152():
    return ResNetSuper(BottleneckSuper, [3,8,36,3],nn.Conv2d,nn.Linear)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


if __name__ == "__main__":
    test()