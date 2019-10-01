import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from utee import misc
print = misc.logger.info

class CaffeLeNet(nn.Module):
    def __init__(self):
        super(CaffeLeNet, self).__init__()
        feature_layers = []
        feature_layers.append(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        )
        feature_layers.append(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        feature_layers.append(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        )
        feature_layers.append(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x