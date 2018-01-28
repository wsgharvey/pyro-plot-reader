import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

import numpy as np

from proposal_layers import proposal_layer


class Administrator(nn.Module):
    """
    keeps track of all sample statement names and instances in program and
    provides appropriate sample-specific layers
    """
    def __init__(self, sample_statements, HYPERPARAMS):
        super(Administrator, self).__init__()
        """
        TODO: consider making it simple to change which layers are shared (e.g. sharing proposal layers between instances)

        example `sample_statements`:

        {"bar_height": {"instances": 3,
                        "dist": dist.uniform},
         "colour": {"instances": 1,
                    "dist": dist.categorical,
                    "n_categories": 5}}     # these special arguments are required for some dists

        """
        self.sample_statements = OrderedDict(sample_statements)
        self.dists = list(set([address["dist"] for address in sample_statements]))
        t_dim = HYPERPARAMS["embedding_size"] + 2*len(self.sample_statements) + 2*len(self.dists)

        self.proposal_layers = nn.ModuleList([nn.ModuleList([proposal_layer(addess["dist"],
                                                                            address[n_categories] if n_categories in address else None)
                                                             for _ in range(address["instances"])])
                                              for address in self.sample_statements])

        self.query_layers = nn.ModuleList([nn.ModuleList([QueryLayer(t_dim,
                                                                     HYPERPARAMS["hidden_size"],
                                                                     HYPERPARAMS["n_queries"],
                                                                     HYPERPARAMS["d_k"])
                                                          for _ in range(address["instances"])])
                                           for address in self.sample_statements])

        self.sample_embedders = nn.ModuleList([nn.ModuleList([SampleEmbedder(1,                              # TODO: shouldn't always be 1
                                                                             HYPERPARAMS["smp_emb_dim"])
                                                              for _ in range address["instances"]])
                                               for address in self.sample_statements])

        self.first_sample_embedding = nn.Parameter(torch.normal(torch.zeros(1, HYPERPARAMS["smp_emb_dim"]), 1))

    def _get_address_index(address):
        return list(self.sample_statements.keys()).index(address)

    def get_proposal_layer(address, instance):
        address_index = self._get_address_index(address)
        return self.proposal_layers[address_index][instance]

    def get_sample_embedder(address, instance):
        address_index = self._get_address_index(address)
        return self.sample_embedders[address_index][instance]

    def get_query_layer(address, instance):
        address_index = self._get_address_index(address)
        return self.query_layers[address_index][instance]

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

    def t(prev_sample_name, current_sample_name, prev_sample_value):
        """
        returns an embedding of current and previous sample statement names and
        distributions and the previously sampled value (a.k.a. t)
        """
        if prev_sample_name is None:
            t = torch.cat([Variable(torch.zeros(1, len(self.sample_statements))),
                           Variable(torch.zeros(1, len(self.dists))),
                           self.one_hot_address(current_sample_name),
                           self.one_hot_distribution(current_sample_name),
                           self.first_sample_embedding], 1)
        else:
            t = torch.cat([self.one_hot_address(prev_sample_name),
                           self.one_hot_distribution(prev_sample_name),
                           self.one_hot_address(current_sample_name),
                           self.one_hot_distribution(current_sample_name),
                           self.get_sample_embedder(prev_sample_name)(prev_sample_value)], 1)
        return t


# TODO: move SampleEmbedder and QueryLayer
class SampleEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.fcn = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(1, -1)
        return F.relu(self.fcn(x))


class QueryLayer(nn.Module):
    def __init__(self, t_dim, hidden_size, n_queries, d_k):
        super(QueryLayer, self).__init__()
        self.n_queries = n_queries
        self.d_k = d_k
        self.fcn1 = nn.Linear(t_dim+hidden_size, n_queries*d_k)
        self.fcn2 = nn.Linear(n_queries*d_k, n_queries*d_k)

    def forward(self, t, prev_hidden):
        t = t.view(1, -1)
        prev_hidden = prev_hidden.view(1, -1)
        x = torch.cat((t, prev_hidden), 1)
        x = self.fcn2(F.relu(self.fcn1(x)))
        x = x.view(self.n_queries, self.d_k)
        return x
