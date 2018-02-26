import torch
import torch.nn as nn


class HaltingUnit(nn.Module):
    def __init__(self, input_dim):
        super(HaltingUnit, self).__init__()
        self.lin = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.lin(x)).view(1)
