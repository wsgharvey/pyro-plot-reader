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
        # RNN to recurrently propose attention location weights
        # - input is the avg_pooled image
        # - outputs a ~20x20 mask of attention weights

        # a couple of padded convs
        # and then strided + not padded conv to reduce to ~20x20

        # then < dot product > with attention weights to get a vector

        # then fcn this shit

        self.low_res_pool = nn.AvgPool2d(10, stride=10)    # input to "lstm"

        # Internals of "lstm":
        self.img_conv1 = nn.Conv2d(3, 5, 3, padding=1)
        self.img_conv2 = nn.Conv2d(5, 10, 3, padding=1)
        self.img_conv3 = nn.Conv2d(10, 10, 3, padding=1)
        self.att_conv1 = nn.Conv2d(1, 5, 3, padding=1)
        self.att_conv2 = nn.Conv2d(5, 10, 3, padding=1)
        self.att_conv3 = nn.Conv2d(10, 10, 3, padding=1)
        # then add these layers together and do more convs
        self.full_conv1 = nn.Conv2d(10, 10, 3, padding=1)
        self.full_conv2 = nn.Conv2d(10, 5, 3, padding=1)
        self.full_conv3 = nn.Conv2d(5, 1, 3, padding=1)

        # apply these to full res image
        self.conv1 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(5, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)
        self.conv4 = nn.Conv2d(20, 40, 3, padding=1)
        self.pool = nn.MaxPool2d(5, stride=5)
        self.conv5 = nn.Conv2d(40, 40, 2, stride=2)
        # then dot with attention weights and do this
        self.fcn1 = nn.Linear(40, 5)
        self.fcn2 = nn.Linear(5, 1)

        self.log_std = nn.Parameter(torch.Tensor([1]))

    def forward(self, observed_image=None):
        assert observed_image is not None
        x = observed_image.view(1, 3, 200, 200)

        std = self.log_std.exp()

        # get image embedding for attention weights calculator
        low_res = self.low_res_pool(x)
        low_res_emb = F.relu(self.img_conv1(low_res))
        low_res_emb = F.relu(self.img_conv2(low_res_emb))
        low_res_emb = self.img_conv3(low_res_emb)

        # get image embedding for calculating heights
        img_emb = F.relu(self.conv1(x))
        img_emb = F.relu(self.conv2(img_emb))
        img_emb = F.relu(self.conv3(img_emb))
        img_emb = F.relu(self.conv4(img_emb))
        img_emb = self.pool(img_emb)
        img_emb = F.relu(self.conv5(img_emb))

        attention_weights = Variable(torch.zeros(1, 1, 20, 20))

        for bar_num in range(3):
            # Calculate next attention weights
            attention_emb = F.relu(self.att_conv1(attention_weights))
            attention_emb = F.relu(self.att_conv2(attention_emb))
            attention_emb = self.att_conv3(attention_emb)
            h = attention_emb + low_res_emb
            h = F.relu(self.full_conv1(h))
            h = F.relu(self.full_conv2(h))
            attention_weights = self.full_conv3(h)

            # Use attention weights to do the actual thing
            # img_emb: (1, 40, 20, 20)
            # attention_weights: (1, 1, 20, 20)
            local_emb = sum(img_emb[0, :, i, j]*attention_weights[0, 0, i, j]
                            for i in range(20)
                            for j in range(20))
            local_emb = local_emb.view(1, 40)

            mean = self.fcn2(F.relu(self.fcn1(local_emb)))

            mean = mean.view(-1)
            mean = mean[0]

            pyro.sample("bar_height_{}".format(bar_num),
                        dist.normal,
                        mean,
                        std)
