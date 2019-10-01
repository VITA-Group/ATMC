import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    rank_accumulator = 0
    def __init__(self, in_planes, out_planes, stride, ConvF, ranks=None, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.ranks=ranks
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        if ConvF is nn.Conv2d:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        else:
            self.conv1 = ConvF(in_planes, out_planes, kernel_size=3, stride=stride, rank=ranks[self.rank_accumulator],
                                padding=1, bias=False)
            self.rank_accumulator += 1
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        if ConvF is nn.Conv2d:
            # print("nn.Conv2d, out_planes {}".format(out_planes))
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        else:
            # print("ConvF, out_planes {}".format(out_planes))
            self.conv2 = ConvF(out_planes, out_planes, kernel_size=3, stride=1, rank=ranks[self.rank_accumulator],
                                padding=1, bias=False)
            self.rank_accumulator += 1
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        if ConvF is nn.Conv2d:
            self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        else:
            self.convShortcut = (not self.equalInOut) and ConvF(in_planes, out_planes, kernel_size=1, stride=stride, 
                                padding=0, bias=False, rank=ranks[self.rank_accumulator]) or None
            if self.convShortcut:
                self.rank_accumulator += 1

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        # print("first out {}".format(out.shape))
        out = self.conv2(out)
        # print(self.ranks)
        # print(self.conv2)
        # print("second out {}".format(out.shape))
        # print(x.shape)
        # print( self.convShortcut and self.convShortcut(x).shape)
        # print(self.equalInOut)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    rank_accumulator = 0
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, ConvF, dropRate=0.0, ranks=None):
        super(NetworkBlock, self).__init__()
        # print("what should be {}".format(ConvF))
        self.ConvF = ConvF
        self.ranks = ranks
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        if self.ConvF is nn.Conv2d:
            for i in range(int(nb_layers)):
                layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, ConvF=self.ConvF, dropRate=dropRate))
        else:
            for i in range(int(nb_layers)):
                block_unit = block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, ConvF=self.ConvF, ranks=self.ranks[self.rank_accumulator:], dropRate=dropRate)
                self.rank_accumulator += block_unit.rank_accumulator
                layers.append(block_unit)
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNetSuper(nn.Module):
    rank_accumulator = 0
    def __init__(self, depth, num_classes, ConvF, LinearF, ranks=None, widen_factor=1, dropRate=0.0):
        super(WideResNetSuper, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        if ConvF is nn.Conv2d:
            self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
            # 1st block
            self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, ConvF=ConvF, dropRate=dropRate)
            # 2nd block
            self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, ConvF=ConvF, dropRate=dropRate)
            # 3rd block
            self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, ConvF=ConvF, dropRate=dropRate)
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(nChannels[3])
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(nChannels[3], num_classes)
            self.nChannels = nChannels[3]
        else:
            self.conv1 = ConvF(3, nChannels[0], kernel_size=3, stride=1, rank=ranks[self.rank_accumulator],
                               padding=1, bias=False)
            self.rank_accumulator += 1
            # 1st block
            self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, ConvF=ConvF, ranks=ranks[self.rank_accumulator:], dropRate=dropRate)
            self.rank_accumulator += self.block1.rank_accumulator
            # 2nd block
            self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, ConvF=ConvF, ranks=ranks[self.rank_accumulator:], dropRate=dropRate)
            self.rank_accumulator += self.block2.rank_accumulator
            # 3rd block
            self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, ConvF=ConvF, ranks=ranks[self.rank_accumulator:], dropRate=dropRate)
            self.rank_accumulator += self.block3.rank_accumulator
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(nChannels[3])
            self.relu = nn.ReLU(inplace=True)
            self.fc = LinearF(nChannels[3], num_classes, rank=ranks[self.rank_accumulator])
            self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

if __name__ == "__main__":
    net = WideResNetSuper(depth=16, num_classes=10, ConvF=nn.Conv2d, LinearF=nn.Linear, ranks=None, widen_factor=8, dropRate=0.4)
    y = net(torch.randn(1,3,32,32))
    print(y.size())