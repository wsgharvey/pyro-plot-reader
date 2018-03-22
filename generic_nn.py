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
                    "output_dim": 5}}     # output_dim is required if not 1

        """
        self.HYPERPARAMS = HYPERPARAMS
        self.sample_statements = OrderedDict(sample_statements)
        self.dists = list(set([address["dist"] for address in sample_statements.values()]))
        self.max_instances = max(v["instances"] for v in sample_statements.values())
        self.t_dim_with_low_res = HYPERPARAMS["smp_emb_dim"] + 2*len(self.sample_statements) + 2*len(self.dists) + self.max_instances + self.HYPERPARAMS["low_res_emb_size"]
        self.t_dim_without_low_res = HYPERPARAMS["smp_emb_dim"] + 2*len(self.sample_statements) + 2*len(self.dists) + self.max_instances

        if HYPERPARAMS["share_prop_layer"]:
            self.proposal_layers = nn.ModuleList([proposal_layer(address["dist"],
                                                                 HYPERPARAMS["hidden_size"],
                                                                 address["output_dim"] if "output_dim" in address else 1)
                                                  for address in self.sample_statements.values()])
        else:
            self.proposal_layers = nn.ModuleList([nn.ModuleList([proposal_layer(address["dist"],
                                                                                HYPERPARAMS["hidden_size"],
                                                                                address["output_dim"] if "output_dim" in address else 1)
                                                                 for _ in range(address["instances"])])
                                                  for address in self.sample_statements.values()])

        self.transform_layers = nn.ModuleList([nn.ModuleList([TransformLayer(self.t_dim_with_low_res,
                                                                         HYPERPARAMS["hidden_size"])
                                                              for _ in range(address["instances"])])
                                               for address in self.sample_statements.values()])

        if HYPERPARAMS["share_smp_embedder"]:
            self.sample_embedders = nn.ModuleList([SampleEmbedder(1, HYPERPARAMS["smp_emb_dim"])
                                                   for address in self.sample_statements.values()])
        else:
            self.sample_embedders = nn.ModuleList([nn.ModuleList([SampleEmbedder(1,                              # TODO: shouldn't always be 1
                                                                                 HYPERPARAMS["smp_emb_dim"])
                                                                  for _ in range(address["instances"])])
                                                   for address in self.sample_statements.values()])

        self.first_sample_embedding = nn.Parameter(torch.normal(torch.zeros(1, HYPERPARAMS["smp_emb_dim"]), 1))

    def _get_address_index(self, address):
        return list(self.sample_statements.keys()).index(address)

    def get_proposal_layer(self, address, instance):
        address_index = self._get_address_index(address)
        if self.HYPERPARAMS["share_prop_layer"]:
            return self.proposal_layers[address_index]
        else:
            return self.proposal_layers[address_index][instance]

    def get_sample_embedder(self, address, instance):
        address_index = self._get_address_index(address)
        if self.HYPERPARAMS["share_smp_embedder"]:
            return self.sample_embedders[address_index]
        else:
            return self.sample_embedders[address_index][instance]

    def get_transform_layer(self, address, instance):
        address_index = self._get_address_index(address)
        return self.transform_layers[address_index][instance]

    def one_hot_address(self, address):
        one_hot = Variable(torch.zeros(1, len(self.sample_statements)))
        if self.HYPERPARAMS["CUDA"]:
            one_hot = one_hot.cuda()
        address_index = self._get_address_index(address)
        one_hot[0, address_index] = 1
        return one_hot

    def one_hot_distribution(self, address):
        one_hot = Variable(torch.zeros(1, len(self.dists)))
        if self.HYPERPARAMS["CUDA"]:
            one_hot = one_hot.cuda()
        dist_index = self.dists.index(self.sample_statements[address]["dist"])
        one_hot[0, dist_index] = 1
        return one_hot

    def one_hot_instance(self, instance):
        one_hot = Variable(torch.zeros(1, self.max_instances))
        if self.HYPERPARAMS["CUDA"]:
            one_hot = one_hot.cuda()
        one_hot[0, instance] = 1
        return one_hot

    def t(self,
          current_instance,
          current_sample_name,
          prev_instance,
          prev_sample_name,
          prev_sample_value,
          low_res_emb=None):
        """
        returns an embedding of current and previous sample statement names and
        distributions and the previously sampled value (a.k.a. t)
        """
        if prev_sample_name is None:
            no_address = Variable(torch.zeros(1, len(self.sample_statements)))
            no_distribution = Variable(torch.zeros(1, len(self.dists)))
            if self.HYPERPARAMS["CUDA"]:
                no_address = no_address.cuda()
                no_distribution = no_distribution.cuda()
            t = torch.cat([no_address,
                           no_distribution,
                           self.one_hot_address(current_sample_name),
                           self.one_hot_distribution(current_sample_name),
                           self.one_hot_instance(current_instance),
                           self.first_sample_embedding], 1)
        else:
            if self.HYPERPARAMS["CUDA"]:
                prev_sample_value = prev_sample_value.cuda()
            t = torch.cat([self.one_hot_address(prev_sample_name),
                           self.one_hot_distribution(prev_sample_name),
                           self.one_hot_address(current_sample_name),
                           self.one_hot_distribution(current_sample_name),
                           self.one_hot_instance(current_instance),
                           self.get_sample_embedder(prev_sample_name, prev_instance)(prev_sample_value)], 1)
        if low_res_emb is not None:
            t = torch.cat([t, low_res_emb], 1)
        return t

        @staticmethod
        def remove_low_res_emb(t):
            low_res_emb_size = self.HYPERPARAMS["low_res_emb_size"]
            return t[:-low_res_emb_size]


# TODO: move SampleEmbedder and QueryLayer
class SampleEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SampleEmbedder, self).__init__()
        self.fcn = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(1, -1)
        return F.relu(self.fcn(x))


class TransformLayer(nn.Module):
    def __init__(self, t_dim_with_low_res, hidden_size):
        super(TransformLayer, self).__init__()
        self.fcn1 = nn.Linear(t_dim_with_low_res+hidden_size, 64)
        self.fcn2 = nn.Linear(64, 6)

        # set transform to be identity transform at first
        self.fcn2.weight.data.fill_(0)
        self.fcn2.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, t, prev_hidden):
        prev_hidden = prev_hidden[-1, :, :]
        t = t.view(1, -1)
        prev_hidden = prev_hidden.view(1, -1)
        x = torch.cat((t, prev_hidden), 1)
        theta = self.fcn2(F.relu(self.fcn1(x)))
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((1, 3, 210, 210)))

        return grid, theta


class TransformEmbedder(nn.Module):
    def __init__(self, output_dim):
        super(TransformEmbedder, self).__init__()
        self.fcn1 = nn.Linear(6, 32)
        self.fcn2 = nn.Linear(32, output_dim)

    def forward(self, theta):
        theta = theta.view(-1, 6)
        x = self.fcn2(F.relu(self.fcn1(theta)))
        return x
