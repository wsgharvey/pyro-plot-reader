import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np

from attention import FourierLocationEmbedder,\
                      LearnedLocationEmbedder,\
                      MultiHeadAttention
from generic_nn import Administrator
from graphics import AttentionTracker
from cnn import ViewEmbedder, FullViewEmbedder


class Guide(nn.Module):
    def __init__(self,
                 d_k=64,
                 d_emb=128,
                 n_queries=4,
                 hidden_size=1024,
                 lstm_layers=1,
                 smp_emb_dim=32,
                 n_attention_heads=4,
                 lstm_dropout=0.1,
                 use_low_res_view=True,
                 low_res_view_as_attention_loc=False,
                 low_res_emb_size=128,
                 cuda=False,
                 share_smp_embedder=True,
                 share_qry_layer=True,
                 share_prop_layer=True,
                 keys_use_view=True,
                 keys_use_loc=True,
                 vals_use_view=True,
                 vals_use_loc=True,
                 learn_loc_embs=False,
                 wiggle_picture=False,
                 max_loc_emb_wiggle=0,
                 add_linear_loc_emb=True,
                 random_colour=True,
                 random_bar_width=True,
                 random_line_colour=True,
                 random_line_width=True,
                 attention_graphics_path=None,
                 collect_history=False):

        super(Guide, self).__init__()

        self.HYPERPARAMS = {"d_k": d_k,
                            "d_emb": d_emb,     # cannot be changed without changing FourierLocationEmbedder
                            "n_queries": n_queries,
                            "hidden_size": hidden_size,
                            "lstm_layers": lstm_layers,
                            "smp_emb_dim": smp_emb_dim,
                            "n_attention_heads": n_attention_heads,
                            "lstm_dropout": lstm_dropout,
                            "use_low_res_view": use_low_res_view,
                            "low_res_emb_size": low_res_emb_size,
                            "CUDA": cuda,
                            "share_smp_embedder": share_smp_embedder,
                            "share_qry_layer": share_qry_layer,
                            "share_prop_layer": share_prop_layer,
                            "keys_use_view": keys_use_view,
                            "keys_use_loc": keys_use_loc,
                            "vals_use_view": vals_use_view,
                            "vals_use_loc": vals_use_loc,
                            "max_loc_emb_wiggle": max_loc_emb_wiggle}
        self.CUDA = cuda
        self.random_colour = random_colour
        self.random_bar_width = random_bar_width
        self.random_line_colour = random_line_colour
        self.random_line_width = random_line_width
        self.wiggle_picture = wiggle_picture

        self.sample_statements = {"num_bars":   {"instances": 1,
                                                 "dist": dist.categorical,
                                                 "output_dim": 5},
                                  "bar_height": {"instances": 5,
                                                 "dist": dist.uniform}}
        if wiggle_picture:
            shifts = {shift: {"instances": 1,
                              "dist": dist.categorical,
                              "output_dim": 10}
                      for shift in ("x_shift", "y_shift")}
            self.sample_statements.update(shifts)
        if random_colour:
            colour_samples = {col: {"instances": 1,
                                    "dist": dist.uniform}
                              for col in ("red", "green", "blue")}
            self.sample_statements.update(colour_samples)
        if random_bar_width:
            self.sample_statements.update({"bar_width": {"instances": 1,
                                                         "dist": dist.uniform}})
        if random_line_colour:
            colour_samples = {"line_{}".format(col): {"instances": 1,
                                                      "dist": dist.uniform}
                              for col in ("red", "green", "blue")}
            self.sample_statements.update(colour_samples)
        if random_line_width:
            self.sample_statements.update({"line_width": {"instances": 1,
                                                          "dist": dist.uniform}})

        self.administrator = Administrator(self.sample_statements,
                                           self.HYPERPARAMS)

        self.view_embedder = ViewEmbedder(output_dim=d_emb)
        if use_low_res_view:
            self.low_res_embedder = FullViewEmbedder(output_dim=low_res_emb_size)
        if learn_loc_embs:
            self.location_embedder = LearnedLocationEmbedder(d_emb)
        else:
            self.location_embedder = FourierLocationEmbedder(d_emb, 200, 200, add_linear_loc_emb)
        self.mha = MultiHeadAttention(h=n_attention_heads, d_k=d_k, d_v=d_emb, d_model=d_emb)
        self.initial_hidden = nn.Parameter(torch.normal(torch.zeros(lstm_layers, 1, hidden_size), 1))
        self.initial_cell = nn.Parameter(torch.normal(torch.zeros(lstm_layers, 1, hidden_size), 1))

        lstm_input_size = n_queries*d_emb + self.administrator.t_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size,
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

    def init_lstm(self, keys, values, low_res_emb=None):
        """
        run at the start of each trace
        intialises LSTM hidden state etc.
        """
        self.hidden, self.cell = self.initial_hidden, self.initial_cell
        self.instances_dict = {key: -1 for key in self.sample_statements}   # maybe messy to start from -1
        self.keys = keys
        self.values = values
        self.low_res_emb = low_res_emb
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
                                 prev_sample_value,
                                 self.low_res_emb)
        queries = self.administrator.get_query_layer(current_sample_name, current_instance)(t=t,
                                                                                            prev_hidden=self.hidden)
        if self.attention_tracker is None:
            attention_output = self.mha(queries, self.keys, self.values).view(1, -1)
        else:
            attention_output = self.mha(queries, self.keys, self.values, self.attention_tracker).view(1, -1)

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

        if self.CUDA:
            try:
                proposal_params = proposal_params.cpu()
            except AttributeError:
                proposal_params = tuple(param.cpu() for param in proposal_params)
        return proposal_params

    def forward(self, observed_image=None):
        x = observed_image.view(1, 3, 210, 210)
        if self.CUDA:
            x = x.cuda()

        if self.collect_history:
            self.history.append([])

        if self.HYPERPARAMS["use_low_res_view"]:
            low_res_img = nn.AvgPool2d(10)(x)
            low_res_emb = self.low_res_embedder(low_res_img)

        """ this bit should probably be moved """
        # find and embed each seperate location
        views = [x[:, :, 10*j:10*(j+2), 10*i:10*(i+2)].clone().view(1, 3, 20, 20) for i in range(20) for j in range(20)]
        views = torch.cat(views, 0)
        view_embeddings = self.view_embedder(views)

        # add location embeddings
        x_offset = np.random.uniform(0, self.HYPERPARAMS["max_loc_emb_wiggle"])  # will be 0 by default
        y_offset = np.random.uniform(0, self.HYPERPARAMS["max_loc_emb_wiggle"])  # will be 0 by default
        location_embeddings = [self.location_embedder(i, j, x_offset=x_offset, y_offset=y_offset)
                               for i in range(0, 200, 10)
                               for j in range(0, 200, 10)]
        if isinstance(x.data, torch.cuda.FloatTensor):
            location_embeddings = [emb.cuda() for emb in location_embeddings]
        location_embeddings = torch.cat(location_embeddings, 0)

        keys = Variable(torch.zeros(view_embeddings.shape))
        values = Variable(torch.zeros(view_embeddings.shape))
        if self.CUDA:
            keys = keys.cuda()
            values = values.cuda()
        if self.HYPERPARAMS["keys_use_view"]:
            keys += view_embeddings
        if self.HYPERPARAMS["keys_use_loc"]:
            keys += location_embeddings
        if self.HYPERPARAMS["vals_use_view"]:
            values += view_embeddings
        if self.HYPERPARAMS["vals_use_loc"]:
            values += location_embeddings

        keys = torch.cat([keys, low_res_emb.view(1, -1)], 0)
        values = torch.cat([values, low_res_emb.view(1, -1)], 0)
        # low_res_img = nn.AvgPool2d(10)(observed_image.view(1, 3, 200, 200))
        # full_pic_embedding = self.big_picture_embedder(low_res_img)
        # x = torch.cat([view_embeddings, full_pic_embedding], 0)
        """"""

        if self.HYPERPARAMS["use_low_res_view"]:
            self.init_lstm(keys, values, low_res_emb)
        else:
            self.init_lstm(keys, values, None)
        prev_sample_value = None

        if self.wiggle_picture:
            for shift in ("x_shift", "y_shift"):
                ps = self.time_step(shift, prev_sample_value)
                prev_sample_value = pyro.sample(shift,
                                                proposal_dists.categorical_proposal,
                                                ps=ps).type(torch.FloatTensor)

        if self.random_colour:
            for colour in ("red", "green", "blue"):
                modes, certainties = self.time_step(colour,
                                                    prev_sample_value)
                mode, certainty = modes[0], certainties[0]
                prev_sample_value = pyro.sample(colour,
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_bar_width:
            modes, certainties = self.time_step("bar_width",
                                                prev_sample_value)
            mode, certainty = modes[0], certainties[0]
            prev_sample_value = pyro.sample("bar_width",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([1])),
                                            mode,
                                            certainty)
        if self.random_line_colour:
            for colour in ("red", "green", "blue"):
                modes, certainties = self.time_step("line_{}".format(colour),
                                                    prev_sample_value)
                mode, certainty = modes[0], certainties[0]
                prev_sample_value = pyro.sample("line_{}".format(colour),
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_line_width:
            modes, certainties = self.time_step("line_width",
                                                prev_sample_value)
            mode, certainty = modes[0], certainties[0]
            prev_sample_value = pyro.sample("line_width",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([2.5])),
                                            mode*2.5,
                                            certainty)

        ps = self.time_step("num_bars",
                            prev_sample_value)
        num_bars = pyro.sample("num_bars",
                                proposal_dists.categorical_proposal,
                                ps=ps)
        prev_sample_value = num_bars.type(torch.FloatTensor)

        for _ in range(num_bars):
            modes, certainties = self.time_step("bar_height",
                                                prev_sample_value)
            mode, certainty = modes[0], certainties[0]
            print(mode.data.numpy()[0])
            prev_sample_value = pyro.sample("{}_{}".format("bar_height", self.instances_dict["bar_height"]),
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([10])),
                                            mode*10,
                                            certainty)

        if self.attention_tracker is not None:
            self.attention_tracker.save_graphics()

    def get_history(self):
        if not self.collect_history:
            raise Exception("collect_history is set to False")
        return self.history
