import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPSuper(nn.Module):
    offset = 0
    def __init__(self, ConvF, LinF, ranks=None, quant_forward=False, bit=32):
        super(MLPSuper, self).__init__()

        if (ConvF is nn.Conv2d and LinF is nn.Linear):
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(28 * 28, 300)
            self.fc2 = nn.Linear(300, 100)
            self.fc3 = nn.Linear(100, 10)
        else:
            self.relu = nn.ReLU()
            self.fc1 = LinF(28 * 28, 300, rank=ranks[self.offset])
            self.offset += 1
            self.fc2 = LinF(300, 100, rank=ranks[self.offset])
            self.offset += 1
            self.fc3 = LinF(100, 10, rank=ranks[self.offset])
            self.offset += 1


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)