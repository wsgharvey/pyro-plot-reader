import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

import numpy as np

from proposal_layers import proposal_layer


class SampleStatementContainer(nn.Module):
    def __init__(self, sample_statements,
                 proposal_layer_args={},
                 query_layer_args={},
                 sample_embedder_args={}):
        super(SampleStatementTracker, self).__init__()
        self.sample_statements = OrderedDict(sample_statements)
        self.dists = list(set([address["dist"] for address in sample_statements]))
        self.proposal_layers = nn.ModuleList([nn.ModuleList([proposal_layer(addess["dist"])(hidden_size)
                                                             for _ in range(address["instances"])])
                                              for address in self.sample_statements])
        self.query_layers = nn.ModuleList([nn.ModuleList([QueryLayer(...)
                                                          for _ in range(address["instances"])])
                                           for address in self.sample_statements])
        self.sample_embedders = nn.ModuleList([SampleEmbedder(...) for address in self.sample_statements])

    def _get_address_index(address):
        return list(self.sample_statements.keys()).index(address)

    def get_proposal_layer(address, instance):
        address_index = self._get_address_index(address)
        return self.proposal_layers[address_index][instance]

    def get_sample_embedder(address):
        index = self._get_address_index(address)
        return self.sample_embedders[index]

    def one_hot_address(address):
        one_hot = Variable(torch.zeros(1, len(self.sample_statements)))
        address_index = self._get_address_index(address)
        one_hot[0, address_no] = 1
        return one_hot

    def one_hot_distribution(address):
        one_hot = Variable(torch.zeros(1, len(self.dists)))
        dist_index = self.dists.index(self.sample_statements[address]["dist"])
        one_hot[0, dist_index] = 1
        return one_hot


class SampleEmbedder(nn.Module):
    def __init__(self, input_dim=1, output_dim):    # TODO: change so input_dim isn't always 1
        self.fcn = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fcn(x))
