import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np

from attention import LocationEmbeddingMaker, MultiHeadAttention
from generic_nn import Administrator
from graphics import AttentionTracker


class ViewEmbedder(nn.Module):
    def __init__(self):
        super(ViewEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.conv7 = nn.Conv2d(64, 128, 3)
        self.conv8 = nn.Conv2d(128, 128, 3)
        self.conv9 = nn.Conv2d(128, 256, 3)
        self.conv10 = nn.Conv2d(256, 128, 2)

    def forward(self, x):
        x = x.view(1, 3, 20, 20)
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        x = x.view(1, 128)
        return x


class SampleEmbedder(nn.Module):
    """
    use one of these per sample address
    """
    def __init__(self, output_dim):
        super(SampleEmbedder, self).__init__()
        self.fcn1 = nn.Linear(1, output_dim)
        self.fcn2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = x.view(1, 1)
        return self.fcn2(F.relu(self.fcn1(x)))


class Guide(nn.Module):
    def __init__(self,
                 d_k=64,
                 d_v=128,
                 d_model=128,
                 n_queries=16,
                 hidden_size=2048,
                 lstm_layers=1,
                 smp_emb_dim=32,
                 n_attention_queries=20,
                 n_attention_heads=8,
                 lstm_dropout=0.1,
                 cuda=False,
                 share_smp_embedder=False,
                 share_qry_layer=False,
                 share_prop_layer=False,
                 attention_graphics_path=None,
                 collect_history=False):

        super(Guide, self).__init__()

        self.HYPERPARAMS = {"d_k": d_k,
                            "d_v": d_v,
                            "d_model": d_model,
                            "n_queries": n_queries,
                            "hidden_size": hidden_size,
                            "lstm_layers": lstm_layers,
                            "smp_emb_dim": smp_emb_dim,
                            "n_attention_queries": n_attention_queries,
                            "n_attention_heads": n_attention_heads,
                            "lstm_dropout": lstm_dropout,
                            "CUDA": cuda,
                            "share_smp_embedder": share_smp_embedder,
                            "share_qry_layer": share_qry_layer,
                            "share_prop_layer": share_prop_layer}
        self.CUDA = cuda

        self.sample_statements = {"num_bars": {"instances": 1,
                                               "dist": dist.categorical,
                                               "output_dim": 5},
                                  "bar_height": {"instances": 3,
                                                 "dist": dist.uniform}}
        self.administrator = Administrator(self.sample_statements,
                                           self.HYPERPARAMS)

        self.view_embedder = ViewEmbedder()
        self.location_embedder = LocationEmbeddingMaker(200, 200)
        self.mha = MultiHeadAttention(h=n_attention_heads, d_k=d_k, d_v=d_v, d_model=d_model)   # TODO: get rid of d_model?
        self.initial_hidden = nn.Parameter(torch.normal(torch.zeros(1, lstm_layers, hidden_size), 1))
        self.initial_cell = nn.Parameter(torch.normal(torch.zeros(1, lstm_layers, hidden_size), 1))
        self.lstm = nn.LSTM(input_size=n_queries*d_v + self.administrator.t_dim,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout)

        if attention_graphics_path is not None:
            self.attention_tracker = AttentionTracker(attention_graphics_path)
        else:
            self.attention_tracker = None

        self.collect_history = collect_history
        if collect_history is True:
            self.history = []

        if cuda:
            self.cuda()

    def init_lstm(self, input_embeddings):
        """
        run at the start of each trace
        intialises LSTM hidden state etc.
        """
        self.hidden, self.cell = self.initial_hidden, self.initial_cell
        self.instances_dict = {key: -1 for key in self.sample_statements}   # maybe messy to start from -1
        self.x = input_embeddings
        self.prev_sample_name = None
        self.prev_instance = None

    def time_step(self, current_sample_name, prev_sample_value):
        """
        perform one LSTM time step
        returns proposal parameters for `current_sample_name`
        """
        self.instances_dict[current_sample_name] += 1
        current_instance = self.instances_dict[current_sample_name]

        t = self.administrator.t(current_instance,
                                 current_sample_name,
                                 self.prev_instance,
                                 self.prev_sample_name,
                                 prev_sample_value)
        queries = self.administrator.get_query_layer(current_sample_name, current_instance)(self.hidden, t)   # this should use sample_name not step
        if self.attention_tracker is None:
            attention_output = self.mha(queries, self.x, self.x).view(1, 2048)
        else:
            attention_output = self.mha(queries, self.x, self.x, self.attention_tracker).view(1, 2048)
        lstm_input = torch.cat([attention_output, t], 1).view(1, 1, -1)

        lstm_output, (hidden, cell) = self.lstm(lstm_input, (self.hidden, self.cell))
        del self.hidden
        del self.cell
        self.hidden, self.cell = hidden, cell

        proposal_params = self.administrator.get_proposal_layer(current_sample_name, current_instance)(lstm_output)

        self.prev_sample_name = current_sample_name
        self.prev_instance = current_instance
        if self.collect_history:
            self.history[-1].append((current_sample_name,
                                     lstm_output))

        return proposal_params

    def forward(self, observed_image=None):
        x = observed_image.view(1, 3, 200, 200)
        if self.CUDA:
            x = x.cuda()

        if self.collect_history:
            self.history.append([])

        """ this bit should probably be moved """
        # find and embed each seperate location
        views = (x[:, :, 10*j:10*(j+2), 10*i:10*(i+2)].clone() for i in range(19) for j in range(19))
        view_embeddings = [self.view_embedder(view) for view in views]

        # add location embeddings
        location_embeddings = [self.location_embedder.createLocationEmbedding(i, j)
                               for i in range(0, 190, 10)
                               for j in range(0, 190, 10)]
        if isinstance(x.data, torch.cuda.FloatTensor):
            location_embeddings = [emb.cuda() for emb in location_embeddings]

        x = torch.cat(view_embeddings, 0) + torch.cat(location_embeddings, 0)
        """"""

        self.init_lstm(x)

        ps = self.time_step("num_bars", None)
        if self.CUDA:
            ps = ps.cpu()
        num_bars = pyro.sample("num_bars",
                                proposal_dists.categorical_proposal,
                                ps=ps)

        current_sample_name = "bar_height"
        prev_sample_value = num_bars.type(torch.FloatTensor)
        for _ in range(num_bars):

            modes, certainties = self.time_step(current_sample_name, prev_sample_value)
            mode, certainty = modes[0]*10, certainties[0]
            if self.CUDA:
                mode = mode.cpu()
                certainty = certainty.cpu()
            print(mode.data.numpy()[0])

            prev_sample_value = pyro.sample("{}_{}".format(current_sample_name, self.instances_dict[current_sample_name]),
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([10])),
                                            mode,
                                            certainty)

        if self.attention_tracker is not None:
            self.attention_tracker.save_graphics()

    def get_history(self):
        if not self.collect_history:
            raise Exception("collect_history is set to False")
        return self.history
