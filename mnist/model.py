import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from utee import misc
print = misc.logger.info

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

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

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            torch.nn.init.xavier_uniform(layers['fc{}'.format(i + 1)].weight)
            # layers['fc{}'.format(i + 1)].bias.fill_(0.)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)
        torch.nn.init.xavier_uniform(layers['out'].weight)
        # layers['out'].bias.fill_(0.)
        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class CNN_BASIC(nn.Module):
    def __init__(self, n_class):
         super(CNN_BASIC, self).__init__()
         self.model = nn.Sequential(
            nn.Conv2d(1, 64, (8, 8), (2, 2), padding=4),
            # nn.ELU(),
            nn.ReLU(),
            nn.Conv2d(64, 64 * 2, (6, 6), (2, 2), padding=0),
            # nn.ELU(),
            nn.ReLU(),
            nn.Conv2d(64 * 2, 64 * 2, (5, 5), (1, 1), padding=0),
            # nn.ELU(),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 2, n_class)
         )
         self.reset_parameters()
         print(self.model)

    def reset_parameters(self):
        for layer in self.model:
            if hasattr(layer, 'weight') and layer.weight.dim() > 1:
                print('reset layer weights with keras default')
                torch.nn.init.xavier_uniform_(layer.weight.data)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input):
        # print(input.shape)
        return self.model.forward(input)

class CNN(nn.Module):
    def __init__(self, n_class):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (8, 8), (2, 2), padding=4),
            nn.ELU(),
            nn.Conv2d(64, 64 * 2, (6, 6), (2, 2), padding=0),
            nn.ELU(),
            nn.Conv2d(64 * 2, 64 * 2, (5, 5), (1, 1), padding=0),
            nn.ELU(),
            Flatten(),
        )
        # enc = nn.Linear(64 * 2, 64 * 2)
        # clf = nn.Linear(64, n_class)
        # dec = nn.Linear(64 * 2, 64 * 2)

        # dis1 = nn.Linear(64, 64)
        # dis2 = nn.Linear(64, 64)

        self.reset_parameters()
        # print(self.model)

    def reset_parameters(self):
        for layer in self.model:
            if hasattr(layer, 'weight') and layer.weight.dim() > 1:
                print('reset layer weights with keras default')
                torch.nn.init.xavier_uniform_(layer.weight.data)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input):
        # print(input.shape)
        return self.model.forward(input)

class UAI_PRED(nn.Module):
    def __init__(self, n_class, confusion_r=0.5):
        super(UAI_PRED, self).__init__()
        self.enc = nn.Linear(64 * 2, 64 * 2)
        self.clf = nn.Linear(64, n_class)
        self.dec = nn.Linear(64 * 2, 64 * 2)

        self.dropout = nn.Dropout(p=confusion_r)

        self.reset_parameters()

    def forward(self, x):
        e1 = x[:, :64]
        e2 = x[:, 64:]

        e1p = self.dropout(e1)
        xp = torch.cat([e1p, e2], dim=1)
        xp = self.dec(xp)

        decLoss = torch.norm((xp - x).view(x.size(0), -1), p=2, dim=-1).sum()

        return self.clf(e1), decLoss

    def reset_parameters(self):
        for w in self.modules():
            if hasattr(w, 'weight') and w.weight.dim() > 1:
                print('reset w weight with keras default')
                torch.nn.init.xavier_uniform_(w.weight.data)
                if hasattr(w, 'bias') and w.bias is not None:
                    w.bias.data.zero_()
        # for layer in self.model:
        #     if hasattr(layer, 'weight') and layer.weight.dim() > 1:
        #         print('reset layer weights with keras default')
        #         torch.nn.init.xavier_uniform_(layer.weight.data)
        #         if hasattr(layer, 'bias') and layer.bias is not None:
        #             layer.bias.data.zero_()

class UAI_DISTORTION(nn.Module):
    def __init__(self):
        super(UAI_DISTORTION, self).__init__()
        self.dis1 = nn.Linear(64, 64)
        self.dis2 = nn.Linear(64, 64)

        self.reset_parameters()

    def forward(self, x):
        e1 = x[:, :64]
        e2 = x[:, 64:]

        e2p = self.dis1(e1)
        e1p = self.dis2(e2)

        # e2p = e2p / torch.norm(e2p, p=2, dim=-1, keepdim=True)
        # e1p = e1p / torch.norm(e1p, p=2, dim=-1, keepdim=True)

        dis1Loss = torch.norm((e1 - e1p).view(e1.size(0), -1), p=2, dim=-1).sum()
        dis2Loss = torch.norm((e2 - e2p).view(e1.size(0), -1), p=2, dim=-1).sum()

        return dis1Loss, dis2Loss

    def reset_parameters(self):
        for w in self.modules():
            if hasattr(w, 'weight') and w.weight.dim() > 1:
                print('reset w weight with keras default')
                torch.nn.init.xavier_uniform_(w.weight.data)
                if hasattr(w, 'bias') and w.bias is not None:
                    w.bias.data.zero_()

def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    # model = MLP(input_dims, n_hiddens, n_class)
    model = CNN(n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

