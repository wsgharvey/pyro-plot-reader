import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np


def proposal_layer(dist_type):
    """
    :dist: a Pyro distribution class
    returns a proposal layer appropriate for `dist`
    """
    if dist == dist.uniform:
        return UniformProposalLayer
    if dist == dist.categorical:
        # FUCK ITS UNCLEAR HOW MANY CATEGORIES THERE ARE
        pass


class ProposalLayer(nn.Module):
    def __init__(self):
        super(ProposalLayer, self).__init__()


class UniformProposalLayer(ProposalLayer):
    def __init__(self, input_dim):
        super(UniformProposalLayer, self).__init__()
        self.fcn1 = nn.Linear(input_dim, input_dim)
        self.fcn2 = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fcn2(F.relu(self.fcn1(x)))
        modes = x[:, 0]
        certainties = x[:, 1]
        modes = nn.Sigmoid()(modes)
        certainties = nn.Softplus()(certainties)
        return modes, certainties
