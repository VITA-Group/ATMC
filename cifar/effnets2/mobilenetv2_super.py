import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    rank_accu = 0
    def __init__(self, in_planes, out_planes, expansion, stride, ranks=None, conv=nn.Conv2d):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        if conv is nn.Conv2d:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )
        else:
            self.conv1 = conv(in_planes, planes, kernel_size=1, rank=ranks[self.rank_accu], stride=1, padding=0, bias=False)
            self.rank_accu += 1
            self.bn1 = nn.BatchNorm2d(planes)
            if planes == 1:
                self.conv2 = conv(planes, planes, kernel_size=3, rank=ranks[self.rank_accu], padding=1, bias=False)
                self.rank_accu += 1
            else:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = conv(planes, out_planes, kernel_size=1, rank=ranks[self.rank_accu], stride=1, padding=0, bias=False)
            self.rank_accu += 1
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    conv(in_planes, out_planes, kernel_size=1, rank=ranks[self.rank_accu], stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )
                self.rank_accu += 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out

        return out

class MobileNetV2(nn.Module):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    offset = 0
    def __init__(self, alpha=1.0, num_classes=10, ranks=None, conv=nn.Conv2d, linear=nn.Linear):
        super(MobileNetV2, self).__init__()
        last_channel = int(1280 * alpha) if alpha > 1.0 else 1280
        if ranks is None:
            self.conv1 = nn.Conv2d(3, int(32*alpha), kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(int(32*alpha))
            self.layers = self._make_layers(in_planes=int(32*alpha), alpha=alpha)
            self.conv2 = nn.Conv2d(int(320*alpha), last_channel, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(last_channel)
            self.linear = nn.Linear(last_channel, num_classes)
        else:
            self.conv1 = conv(3, int(32 * alpha), kernel_size=3, rank=ranks[self.offset], stride=1, padding=1, bias=False)
            self.offset += 1
            self.bn1 = nn.BatchNorm2d(int(32 * alpha))
            self.layers = self._make_layers(in_planes=int(32*alpha), alpha=alpha, ranks=ranks[self.offset:], conv=conv)
            self.conv2 = conv(int(320*alpha), last_channel, kernel_size=1, rank=ranks[self.offset], stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(last_channel)
            self.linear = linear(last_channel, num_classes, rank=ranks[-1])

    def _make_layers(self, in_planes, alpha, ranks=None, conv=nn.Conv2d):
        layers = []
        if ranks is None:
            for expansion, out_planes, num_blocks, stride in self.cfg:
                out_planes = int(out_planes*alpha)
                strides = [stride] + [1]*(num_blocks-1)
                for stride in strides:
                    layers.append(Block(in_planes, out_planes, expansion, stride))
                    in_planes = out_planes
        else:
            base = 0
            for expansion, out_planes, num_blocks, stride in self.cfg:
                out_planes = int(out_planes * alpha)
                strides = [stride] + [1] * (num_blocks - 1)
                for stride in strides:
                    bunit = Block(in_planes, out_planes, expansion, stride, ranks=ranks[base:], conv=conv)
                    layers.append(bunit)
                    in_planes = out_planes
                    base += bunit.rank_accu
                    self.offset += bunit.rank_accu

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()