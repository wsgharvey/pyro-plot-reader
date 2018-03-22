import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ViewEmbedder(nn.Module):
    """
    embeds a 3x210x210 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(ViewEmbedder, self).__init__()
        self.output_dim = output_dim
        self.layers = nn.Sequential(
                        nn.Conv2d(3, 16, 3),                        # 208
                        nn.MaxPool2d(4, stride=4),                  # 52
                        nn.ReLU(True),
                        nn.Conv2d(16, 16, 3),                       # 50
                        nn.ReLU(True),
                        nn.Conv2d(16, 32, 3),                       # 48
                        nn.MaxPool2d(2, stride=2),                  # 24
                        nn.ReLU(True),
                        nn.Conv2d(32, 64, 3, padding=1),            # 24
                        nn.MaxPool2d(2, stride=2),                  # 12
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, 3, padding=1),           # 12
                        nn.MaxPool2d(2, stride=2),                  # 6
                        nn.ReLU(True),
                        nn.Conv2d(128, output_dim, 3, padding=1),   # 6
                        nn.MaxPool2d(2, stride=2),                  # 3
                        nn.ReLU(True),
                        nn.Conv2d(output_dim, output_dim, 3),       # 1
                        )

    def forward(self, x):
        x = x.view(1, 3, 210, 210)
        x = self.layers(x)
        x = x.view(1, self.output_dim)
        return x


class FullViewEmbedder(nn.Module):
    """
    embeds a 3x210x210 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(FullViewEmbedder, self).__init__()
        self.output_dim = output_dim
        self.layers = nn.Sequential(
                        nn.Conv2d(3, 16, 3),                        # 208
                        nn.MaxPool2d(4, stride=4),                  # 52
                        nn.ReLU(True),
                        nn.Conv2d(16, 16, 3),                       # 50
                        nn.ReLU(True),
                        nn.Conv2d(16, 32, 3),                       # 48
                        nn.MaxPool2d(2, stride=2),                  # 24
                        nn.ReLU(True),
                        nn.Conv2d(32, 64, 3, padding=1),            # 24
                        nn.MaxPool2d(2, stride=2),                  # 12
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, 3, padding=1),           # 12
                        nn.MaxPool2d(2, stride=2),                  # 6
                        nn.ReLU(True),
                        nn.Conv2d(128, output_dim, 3, padding=1),   # 6
                        nn.MaxPool2d(2, stride=2),                  # 3
                        nn.ReLU(True),
                        nn.Conv2d(output_dim, output_dim, 3),       # 1
                        )

    def forward(self, x):
        x = x.view(1, 3, 210, 210)
        x = self.layers(x)
        x = x.view(1, self.output_dim)
        return x
