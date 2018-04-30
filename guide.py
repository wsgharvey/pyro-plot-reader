import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np

from generic_nn import Administrator, TransformEmbedder, TransformLayer
from graphics import AttentionTracker
from cnn import ViewEmbedder, FullViewEmbedder


class Guide(nn.Module):
    def __init__(self,
                 view_emb_dim=128,
                 hidden_size=1024,
                 lstm_layers=1,
                 smp_emb_dim=32,
                 n_attention_heads=4,
                 lstm_dropout=0.1,
                 use_low_res_view=False,
                 # low res is always used to select transform - this option lets you decide whether to also use it as an addtional input to LSTM
                 low_res_view_as_attention_loc=False,
                 low_res_emb_size=128,
                 transform_emb_size=16,
                 cuda=False,
                 share_smp_embedder=False,
                 share_prop_layer=False,
                 learn_loc_embs=False,
                 wiggle_picture=False,
                 max_loc_emb_wiggle=0,
                 add_linear_loc_emb=True,
                 random_colour=True,
                 random_bar_width=True,
                 random_line_colour=True,
                 random_line_width=True,
                 scale="fixed",
                 attention_graphics_path=None,
                 collect_history=False):

        super(Guide, self).__init__()

        self.HYPERPARAMS = {"hidden_size": hidden_size,
                            "lstm_layers": lstm_layers,
                            "smp_emb_dim": smp_emb_dim,
                            "n_attention_heads": n_attention_heads,
                            "lstm_dropout": lstm_dropout,
                            "use_low_res_view": use_low_res_view,
                            "low_res_emb_size": low_res_emb_size,
                            "CUDA": cuda,
                            "share_smp_embedder": share_smp_embedder,
                            "share_prop_layer": share_prop_layer,
                            "max_loc_emb_wiggle": max_loc_emb_wiggle}
        self.CUDA = cuda
        self.random_colour = random_colour
        self.random_bar_width = random_bar_width
        self.random_line_colour = random_line_colour
        self.random_line_width = random_line_width
        self.wiggle_picture = wiggle_picture
        self.scale = scale

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
        if scale == "discrete":
            self.sample_statements.update({"max_height": {"instances": 1,
                                                          "dist": dist.categorical,
                                                          "output_dim": 3}})
        elif scale == "continuous":
            self.sample_statements.update({"max_height": {"instances": 1,
                                                          "dist": dist.uniform}})
        else:
            assert scale == "fixed", "scale argument given is invalid"

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

        self.view_embedder = ViewEmbedder(output_dim=view_emb_dim)
        self.low_res_embedder = FullViewEmbedder(output_dim=low_res_emb_size)
        self.transform_embedder = TransformEmbedder(output_dim=transform_emb_size)

        self.initial_hidden = nn.Parameter(torch.normal(torch.zeros(lstm_layers, 1, hidden_size), 1))
        self.initial_cell = nn.Parameter(torch.normal(torch.zeros(lstm_layers, 1, hidden_size), 1))

        if use_low_res_view:
            lstm_input_size = transform_emb_size + view_emb_dim + self.administrator.t_dim_with_low_res
        else:
            lstm_input_size = transform_emb_size + view_emb_dim + self.administrator.t_dim_without_low_res
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

    def init_lstm(self, image, low_res_emb):
        """
        run at the start of each trace
        intialises LSTM hidden state etc.
        """
        self.hidden, self.cell = self.initial_hidden, self.initial_cell
        self.instances_dict = {key: -1 for key in self.sample_statements}   # maybe messy to start from -1
        self.image = image
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
        transform_layer = self.administrator.get_transform_layer(current_sample_name, current_instance)
        transform_grid, transform = transform_layer(t=t, prev_hidden=self.hidden)
        attention_output = F.grid_sample(self.image, transform_grid)
        if self.attention_tracker is not None:
            graphic = attention_output.view(3, 210, 210, 1).data.cpu().numpy()
            graphic = graphic * 255/np.amax(graphic)
            graphic = np.concatenate((graphic[0], graphic[1], graphic[2]), axis=2)
            self.attention_tracker.add_graphic(graphic)
        transform_embedding = self.transform_embedder(transform)
        view_embedding = self.view_embedder(attention_output)

        if not self.HYPERPARAMS["use_low_res_view"]:
            t = self.administrator.remove_low_res_emb(t)
        lstm_input = torch.cat([view_embedding, transform_embedding, t], 1).view(1, 1, -1)

        lstm_output, (hidden, cell) = self.lstm(lstm_input, (self.hidden, self.cell))
        del self.hidden
        del self.cell
        self.hidden, self.cell = hidden, cell

        proposal_params = self.administrator.get_proposal_layer(current_sample_name, current_instance)(lstm_output)

        self.prev_sample_name = current_sample_name
        self.prev_instance = current_instance

        if self.CUDA:
            try:
                proposal_params = proposal_params.cpu()
            except AttributeError:
                proposal_params = tuple(param.cpu() for param in proposal_params)

        if self.collect_history:
            self.history[-1]["{}_{}".format(current_sample_name, current_instance)] = proposal_params

        return proposal_params

    def forward(self, observed_image=None):
        image = observed_image.view(1, 3, 210, 210)
        if self.CUDA:
            image = image.cuda()

        if self.collect_history:
            self.history.append({})

        low_res_emb = self.low_res_embedder(image)
        self.init_lstm(image, low_res_emb)
        prev_sample_value = None

        if self.wiggle_picture:
            for shift in ("x_shift", "y_shift"):
                ps = self.time_step(shift, prev_sample_value)
                prev_sample_value = pyro.sample(shift,
                                                proposal_dists.categorical_proposal,
                                                ps=ps).type(torch.FloatTensor)

        max_max_height = 100
        if self.scale == "discrete":
            ps = self.time_step("max_height", prev_sample_value)
            prev_sample_value = pyro.sample("max_height",
                                            proposal_dists.categorical_proposal,
                                            ps=ps).type(torch.FloatTensor)

            max_height = 100

        elif self.scale == "continuous":
            max_max_height = 100
            mode, certainty = self.time_step("max_height", prev_sample_value)
            prev_sample_value = pyro.sample("max_height",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([max_max_height])),
                                            mode*max_max_height,
                                            certainty)
            max_height = prev_sample_value

            max_height = 100
        else:
            max_height = 10


        if self.random_colour:
            for colour in ("red", "green", "blue"):
                mode, certainty = self.time_step(colour,
                                                 prev_sample_value)
                prev_sample_value = pyro.sample(colour,
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_bar_width:
            mode, certainty = self.time_step("bar_width",
                                             prev_sample_value)
            prev_sample_value = pyro.sample("bar_width",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([1])),
                                            mode,
                                            certainty)
        if self.random_line_colour:
            for colour in ("red", "green", "blue"):
                mode, certainty = self.time_step("line_{}".format(colour),
                                                 prev_sample_value)
                # mode, certainty = modes[0], certainties[0]
                prev_sample_value = pyro.sample("line_{}".format(colour),
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_line_width:
            mode, certainty = self.time_step("line_width",
                                             prev_sample_value)
            # mode, certainty = modes[0], certainties[0]
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
            mode, certainty = self.time_step("bar_height",
                                             prev_sample_value)
            print(mode.data.numpy()[0])
            prev_sample_value = pyro.sample("{}_{}".format("bar_height", self.instances_dict["bar_height"]),
                                            proposal_dists.uniform_proposal, 
                                            Variable(torch.Tensor([0])), 
                                            Variable(torch.Tensor([max_height])),
                                            mode*max_height, 
                                            certainty)

        if self.attention_tracker is not None:
            self.attention_tracker.save_graphics()

    def get_history(self):
        if not self.collect_history:
            raise Exception("collect_history is set to False")
        return self.history
