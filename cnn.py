import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ViewEmbedder(nn.Module):
    """
    embeds a 3x20x20 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(ViewEmbedder, self).__init__()
        self.output_dim = output_dim                                    # 20x20
        self.conv1 = nn.Conv2d(3, 8, 3)                                 # 18x18
        self.conv2 = nn.Conv2d(8, 8, 3)                                 # 16x16
        self.conv3 = nn.Conv2d(8, 16, 3)                                # 14x14
        self.conv4 = nn.Conv2d(16, 16, 3)                               # 12x12
        self.conv5 = nn.Conv2d(16, 16, 3)                               # 10x10
        self.conv6 = nn.Conv2d(16, 32, 3)                               # 8x8
        self.conv7 = nn.Conv2d(32, 64, 3)                               # 6x6
        self.conv8 = nn.Conv2d(64, 64, 3)                               # 4x4
        self.conv9 = nn.Conv2d(64, max(64, output_dim), 3)              # 2x2
        self.conv10 = nn.Conv2d(max(64, output_dim), output_dim, 2)     # 1x1

    def forward(self, x):
        x = x.view(-1, 3, 20, 20)
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        x = x.view(-1, self.output_dim)
        return x


class ViewEmbedder2(nn.Module):
    """
    embeds a 3x20x20 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(ViewEmbedder2, self).__init__()
        self.output_dim = output_dim                                    # 20x20
        self.conv1 = nn.Conv2d(3, 8, 3)                                 # 18x18
        self.conv2 = nn.Conv2d(8, 8, 3)                                 # 16x16
        self.conv3 = nn.Conv2d(8, 16, 3)                                # 14x14
        self.conv4 = nn.Conv2d(16, 8, 3)                                # 12x12
        self.fcn1 = nn.Linear(8*12*12, 128)
        self.fcn2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(-1, 3, 20, 20)
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 8*12*12)
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x)
        x = x.view(-1, self.output_dim)
        return x


class FullViewEmbedder(nn.Module):
    """
    embeds a 3x21x21 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(FullViewEmbedder, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, 8, 3)                         # 19x19
        self.conv2 = nn.Conv2d(8, 8, 3)                         # 17x17
        self.conv3 = nn.Conv2d(8, 16, 2)                        # 16x16
        self.pool = nn.MaxPool2d(2, stride=2)                   # 8x8
        self.conv4 = nn.Conv2d(16, 32, 3)                       # 6x6
        self.conv5 = nn.Conv2d(32, 32, 3)                       # 4x4
        self.conv6 = nn.Conv2d(32, output_dim, 3)               # 2x2
        self.conv7 = nn.Conv2d(output_dim, output_dim, 2)       # 1x1

    def forward(self, x):
        x = x.view(1, 3, 21, 21)
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = x.view(1, self.output_dim)
        return x
