from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.random_primitive import RandomPrimitive

import torch.nn as nn
import torch.functional as F


class ConvStack(nn.Module):
    def __init__(self):
        super(ConvStack, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)      # 208
        self.pool1 = nn.MaxPool2d(2, 2)      # 104
        self.conv2 = nn.Conv2d(3, 6, 3)      # 102
        self.pool2 = nn.MaxPool2d(2, 2)      # 51
        self.conv3 = nn.Conv2d(6, 8, 2)      # 50
        self.pool3 = nn.MaxPool2d(2, 2)      # 25
        self.conv4 = nn.Conv2d(8, 16, 2)     # 24
        self.pool4 = nn.MaxPool2d(2, 2)      # 12
        self.conv5 = nn.Conv2d(16, 32, 3)    # 10
        self.pool5 = nn.MaxPool2d(2, 2)      # 5
        self.conv6 = nn.Conv2d(32, 32, 3)    # 3
        self.conv7 = nn.Conv2d(32, 32, 3)    # 1

    def forward(self, image):
        image = image[0] + image[1] + image[2]
        image = image.view(1, 1, 210, 210)
        image = self.conv1(image)
        image = self.pool1(image)
        image = self.conv2(image)
        image = self.pool2(image)
        image = self.conv3(image)
        image = self.pool3(image)
        image = self.conv4(image)
        image = self.pool4(image)
        image = self.conv5(image)
        image = self.pool5(image)
        image = self.conv6(image)
        image = self.conv7(image)
        return image.view(-1)


class ABCDist(Distribution):
    """
    ABC dist - computes difference between two images after collapsing them
    into stacks
    """
    enumerable = True

    def __init__(self, rendered_image, var):
        self.var = var
        self.rendered_image = rendered_image

        # force conv stack to be initialised with certain random seed
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        self.conv_stack = ConvStack()
        torch.set_rng_state(rng_state)

    # @property
    # def batch_shape(self):
    #     return self.v.size()
    #
    # @property
    # def event_shape(self):
    #     return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        return self.rendered_image

    def log_prob(self, baseline_image):
        rendered_stack = self.conv_stack(self.rendered_image)
        baseline_stack = self.conv_stack(baseline_image)
        baseline_stack = baseline_stack
        stack_difference = rendered_stack - baseline_stack
        approx_pdf = -torch.dot(stack_difference, stack_difference)/self.var
        return approx_pdf

    # def enumerate_support(self, v=None):
    #     """
    #     Returns the delta distribution's support, as a tensor along the first dimension.
    #
    #     :param v: torch variable where each element of the tensor represents the point at
    #         which the delta distribution is concentrated.
    #     :return: torch variable enumerating the support of the delta distribution.
    #     :rtype: torch.autograd.Variable.
    #     """
    #     return Variable(self.v.data.unsqueeze(0))


abc_dist = RandomPrimitive(ABCDist)
