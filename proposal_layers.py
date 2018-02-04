import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np


def proposal_layer(dist_type, input_dim, *args):  # take in arguments and initialise it inside this function ?
    """
    :dist: a Pyro distribution class
    returns a proposal layer appropriate for `dist`
    """
    if dist_type == dist.uniform:
        return UniformProposalLayer(input_dim)
    if dist_type == dist.categorical:
        return CategoricalProposalLayer(input_dim, *args)


class ProposalLayer(nn.Module):
    def __init__(self):
        super(ProposalLayer, self).__init__()


class UniformProposalLayer(ProposalLayer):
    def __init__(self, input_dim):
        super(UniformProposalLayer, self).__init__()
        intermediate_dim = 2*int(input_dim**0.5)
        self.fcn1 = nn.Linear(input_dim, intermediate_dim)
        self.fcn2 = nn.Linear(intermediate_dim, 2)

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fcn2(F.relu(self.fcn1(x)))
        modes = x[:, 0]
        certainties = x[:, 1]
        modes = nn.Sigmoid()(modes)
        certainties = nn.Softplus()(certainties)
        return modes, certainties


class CategoricalProposalLayer(ProposalLayer):
    def __init__(self, input_dim, n_categories):
        super(CategoricalProposalLayer, self).__init__()
        self.fcn1 = nn.Linear(input_dim, int((input_dim*n_categories)**0.5))
        self.fcn2 = nn.Linear(int((input_dim*n_categories)**0.5), n_categories)

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fcn2(F.relu(self.fcn1(x)))
        x = nn.Softmax()(x)
        x = x.view(-1)
        return x
