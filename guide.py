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
    takes in low res image, returns attention weight matrix
    """
    def __init__(self):
        super(AttentionBox, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(10, 5, 3, padding=1)
        self.fcn1 = nn.Linear(500, 50)
        self.fcn2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(1, 3, 20, 20)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(1, 500)
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x)
        x = nn.Softmax()(x)
        x = x.view(1, 10)
        return x


class FocusBox(nn.Module):
    def __init__(self):
        super(FocusBox, self).__init__()
        self.conv1 = nn.Conv2d(40, 40, 3)
        self.conv2 = nn.Conv2d(40, 60, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(60, 60, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(60, 40, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(40, 20, 2)
        self.fcn1 = nn.Linear(20*23, 10)
        self.fcn2 = nn.Linear(10, 1)

    def forward(self, x):
        """
        x is 1x40x200x20
        """
        x = x.view(1, 40, 200, 20)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))

        x = x.view(1, 20*23)
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x)

        return x.view(1)


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        # for calculating attention weights
        self.pool = nn.AvgPool2d(10, stride=10)
        self.attention_boxes = [AttentionBox() for _ in range(3)]
        self.focus_boxes = [FocusBox() for _ in range(10)]

        self.full_conv1 = nn.Conv2d(3, 20, 3, padding=1)
        self.full_conv2 = nn.Conv2d(20, 40, 3, padding=1)
        self.full_conv3 = nn.Conv2d(40, 40, 3, padding=1)
        self.full_conv4 = nn.Conv2d(40, 40, 3, padding=1)

        self.log_std = nn.Parameter(torch.Tensor([1]))

    def forward(self, observed_image=None):
        assert observed_image is not None
        img = observed_image.view(1, 3, 200, 200)
        low_res_img = self.pool(img)

        img_emb = F.relu(self.full_conv1(img))
        img_emb = F.relu(self.full_conv2(img_emb))
        img_emb = F.relu(self.full_conv3(img_emb))
        img_emb = self.full_conv4(img_emb)

        predictions = []
        for i in range(10):
            start, end = i*20, (i+1)*20
            img_slice = img_emb[:, :, :, start:end]
            img_slice.contiguous()
            slice_pred = self.focus_boxes[i](img_slice)
            predictions.append(slice_pred)

        # create 'image' of coordinate values
        # loc_emb = Variable(torch.Tensor(np.array([[[i for i in np.arange(25.)] for j in np.arange(25.)], [[j for i in np.arange(25.)] for j in np.arange(25.)]])))
        # loc_emb = loc_emb.view(1, 2, 25, 25)

        # loc_emb = F.relu(self.loc1(loc_emb))
        # loc_emb = F.relu(self.loc2(loc_emb))
        # loc_emb = self.loc3(loc_emb)

        # full_emb = torch.cat((img_emb, loc_emb), 1)

        # full_emb = F.relu(self.joint1(full_emb))
        # full_emb = F.relu(self.joint2(full_emb))
        # full_emb = self.joint3(full_emb)
        # full_emb = full_emb.view(-1)
        # this will be multiplied with multiple sets of weights so be careful "the troubles" don't happen again

        std = self.log_std.exp()

        for bar_num in range(3):
            attention_weights = self.attention_boxes[bar_num](low_res_img)
            attention_weights = attention_weights.view(-1)
            # now dot the weights with the embedding
            mean = sum(w*x for w, x in zip(attention_weights, predictions))
            print(mean.data.numpy()[0])
            pyro.sample("bar_height_{}".format(bar_num),
                        dist.normal,
                        mean,
                        std)
