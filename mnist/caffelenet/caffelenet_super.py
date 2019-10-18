import torch
import torch.nn as nn
import torch.nn.functional as F
from super_module.super_class import Linear_QuantForward, Conv2d_QuantForward

class CaffeLeNetSuper(nn.Module):
    # expansion = 1
    # rank_accumulator = 0
    offset = 0
    def __init__(self, ConvF, LinF, ranks=None, quant_forward=False, bit=32):
        super(CaffeLeNetSuper, self).__init__()
        feature_layers = []
        if (ConvF is nn.Conv2d and LinF is nn.Linear):
            feature_layers.append(
                ConvF(in_channels=1, out_channels=20, kernel_size=5)
            )
            feature_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            feature_layers.append(
                ConvF(in_channels=20, out_channels=50, kernel_size=5)
            )
            feature_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.features = nn.Sequential(*feature_layers)
            self.classifier = nn.Sequential(
                LinF(50 * 4 * 4, 500),
                nn.ReLU(inplace=True),
                LinF(500, 10)
            )
        elif quant_forward:
            feature_layers.append(
                ConvF(in_channels=1, out_channels=20, kernel_size=5, init_bit=bit)
            )
            feature_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            feature_layers.append(
                ConvF(in_channels=20, out_channels=50, kernel_size=5, init_bit=bit)
            )
            feature_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.features = nn.Sequential(*feature_layers)
            self.classifier = nn.Sequential(
                LinF(50 * 4 * 4, 500, init_bit=bit),
                nn.ReLU(inplace=True),
                LinF(500, 10, init_bit=bit)
            )
        else:
            feature_layers.append(
                ConvF(1, 20, kernel_size=5, rank=ranks[0])
            )
            self.offset += 1
            feature_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            feature_layers.append(
                ConvF(in_channels=20, out_channels=50, kernel_size=5, rank=ranks[self.offset])
            )
            self.offset += 1
            feature_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.features = nn.Sequential(*feature_layers)
            self.classifier = nn.Sequential(
                LinF(50 * 4 * 4, 500, rank=ranks[self.offset]),
                nn.ReLU(inplace=True),
                LinF(500, 10, rank=ranks[-1])
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

def test():
    net = CaffeLeNetSuper(ConvF=nn.Conv2d, LinF=nn.Linear)
    y = net(torch.randn(1, 1, 28, 28))
    print(y.size())

if __name__ == "__main__":
    test()