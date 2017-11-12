import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist


class BarHeightNet(nn.Module):
    def __init__(self):
        super(BarHeightNet, self).__init__()
        self.pool = nn.MaxPool2d(5, stride=5)
        self.conv = nn.Conv2d(3, 1, 1)  # blends the three RGB layers together
        self.fcn1 = nn.Linear(1600, 10)
        self.fcn2 = nn.Linear(10, 1)

    def forward(self, images):
        x = images.view(-1, 3, 200, 200)
        x = self.pool(x)
        x = F.relu(self.conv(x))
        x = x.view(-1, 1600)
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x)
        sig = nn.Sigmoid()
        x = 10 * sig(x)
        x = x.view(-1)
        print("mean height predicted as", x.data.numpy()[0])
        return x
