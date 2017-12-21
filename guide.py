import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist

import numpy as np


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.pool = nn.AvgPool2d(10, stride=10)
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 1, 3)
        self.fcn1 = nn.Linear(256, 1)

        self.log_std = nn.Parameter(torch.Tensor([1]))

    def forward(self, observed_image=None):
        assert observed_image is not None
        x = observed_image.view(1, 3, 200, 200)

        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, 256)
        x = self.fcn1(x)

        mean = x.view(-1)
        std = self.log_std.exp()

        pyro.sample("bar_height",
                    dist.normal,
                    mean,
                    std)
