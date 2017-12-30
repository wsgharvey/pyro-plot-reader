import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist

import numpy as np


class AttentionBox(nn.Module):
    """
    Takes in image + embedding of stuff and outputs regions to attend to
    """
    def __init__(self):
        super(AttentionBox, self).__init__()

    def forward(self, x):
        return nn.AvgPool2d(5, stride=5)(x)


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()

        self.attention_boxes = [AttentionBox() for _ in range(3)]

        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 1, 5)

        self.fcn1s = [nn.Linear(32*32, 40) for _ in range(3)]
        self.fcn2s = [nn.Linear(40, 10) for _ in range(3)]
        self.fcn3s = [nn.Linear(10, 1) for _ in range(3)]

        self.log_std = nn.Parameter(torch.Tensor([1]))

    def forward(self, observed_image=None):
        assert observed_image is not None

        std = self.log_std.exp()

        for bar_num in range(3):
            x = observed_image.view(1, 3, 200, 200)
            x = self.attention_boxes[bar_num](x)

            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            x = x.view(1, 32*32)
            x = F.relu(self.fcn1s[bar_num](x))
            x = F.relu(self.fcn2s[bar_num](x))
            x = self.fcn3s[bar_num](x)
            mean = x.view(1)

            print("predicted:", mean.data.numpy()[0], "with std", std.data.numpy()[0])
            bar_height = pyro.sample("bar_height_{}".format(bar_num),
                                     dist.normal,
                                     mean,
                                     std)
