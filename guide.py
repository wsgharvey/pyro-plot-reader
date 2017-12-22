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
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.conv2 = nn.Conv2d(5, 5, 3)
        self.conv3 = nn.Conv2d(5, 10, 3)
        self.conv4 = nn.Conv2d(10, 10, 3)
        self.conv5 = nn.Conv2d(10, 10, 3)
        self.conv6 = nn.Conv2d(10, 5, 3)
        self.fcn = nn.Linear(320, 3)

        self.log_std = nn.Parameter(torch.Tensor([1]))

    def forward(self, observed_image=None):
        assert observed_image is not None
        x = observed_image.view(1, 3, 200, 200)

        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(-1, 320)
        x = self.fcn(x)

        means = x.view(-1)
        std = self.log_std.exp()

        print("predicted:", means.data.numpy())

        for bar_num in range(3):
            pyro.sample("bar_height_{}".format(bar_num),
                        dist.normal,
                        means[bar_num],
                        std)
