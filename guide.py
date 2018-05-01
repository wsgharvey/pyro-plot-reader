import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np

from generic_nn import Administrator


class CNNEmbedder(nn.Module):
    def __init__(self, d_emb):
        super(CNNEmbedder, self).__init__()
        self.d_emb = d_emb

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)

        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv9 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv11 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3)

        self.conv13 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv14 = nn.Conv2d(512, d_emb, 3)

    def forward(self, x):
        x = x.view(1, 3, 210, 210)          # 210x210
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(2, stride=2)(x)  # 105x105
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = nn.MaxPool2d(2, stride=2, padding=1)(x)  # 53x53
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = nn.MaxPool2d(2, stride=2, padding=1)(x)  # 27x27
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = nn.MaxPool2d(2, stride=2)(x)  # 14x14
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = nn.MaxPool2d(2, stride=2, padding=1)(x)  # 7x7
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = nn.MaxPool2d(2, stride=2, padding=1)(x)  # 3x3
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = x.view(-1, self.d_emb)
        return x


class CNNEmbedder2(nn.Module):
    """
    CNN embedder that doesn't use max pools
    """
    def __init__(self, d_emb):
        super(CNNEmbedder2, self).__init__()
        self.d_emb = d_emb

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)

        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, stride=2)

        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1, stride=2)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1, stride=2)

        self.conv9 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, 3, padding=1, stride=2)

        self.conv11 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, stride=2)

        self.conv13 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv14 = nn.Conv2d(512, d_emb, 3)

    def forward(self, x):
        x = x.view(1, 3, 210, 210)          # 210x210
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))           # 105x105

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))           # 53x53

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))           # 27x27

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))           # 14x14

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))          # 7x7

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))          # 3x3

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        x = x.view(-1, self.d_emb)
        return x


class CNNEmbedder3(nn.Module):
    """
    CNN embedder with more parameters (about 15m)
    """
    def __init__(self, d_emb):
        super(CNNEmbedder3, self).__init__()
        self.d_emb = d_emb

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1, stride=2)

        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1, stride=2)

        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1, stride=2)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=2)

        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv14 = nn.Conv2d(512, d_emb, 3)

    def forward(self, x):
        x = x.view(1, 3, 210, 210)          # 210x210
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))           # 105x105

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))           # 53x53

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))           # 27x27

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))           # 14x14

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))          # 7x7

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))          # 3x3

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        x = x.view(-1, self.d_emb)
        return x


class Guide(nn.Module):
    def __init__(self,
                 d_emb=128,
                 hidden_size=1024,
                 lstm_layers=1,
                 smp_emb_dim=32,
                 lstm_dropout=0.1,
                 cuda=False,
                 share_smp_embedder=False,
                 share_prop_layer=False,
                 wiggle_picture=False,
                 random_colour=True,
                 random_bar_width=True,
                 random_line_colour=True,
                 random_line_width=True,
                 cnn_number=3,
                 scale="fixed",
                 multi_bar_charts=False,
                 random_img_dim=False,
                 random_layout=False,
                 attention_graphics_path=None,
                 collect_history=False):

        super(Guide, self).__init__()

        self.HYPERPARAMS = {"d_emb": d_emb,
                            "hidden_size": hidden_size,
                            "lstm_layers": lstm_layers,
                            "smp_emb_dim": smp_emb_dim,
                            "lstm_dropout": lstm_dropout,
                            "CUDA": cuda,
                            "share_smp_embedder": share_smp_embedder,
                            "share_prop_layer": share_prop_layer}
        self.CUDA = cuda
        self.random_colour = random_colour
        self.random_bar_width = random_bar_width
        self.random_line_colour = random_line_colour
        self.random_line_width = random_line_width
        self.wiggle_picture = wiggle_picture
        self.scale = scale
        self.multi_bar_charts = multi_bar_charts
        self.random_img_dim = random_img_dim
        self.random_layout = random_layout

        self.sample_statements = {"num_bars": {"instances": 1,
                                               "dist": dist.categorical,
                                               "output_dim": 5}}
        if multi_bar_charts:
            self.sample_statements.update({"bar_height_{}".format(bar_chart): {"instances": 5,
                                                                               "dist": dist.uniform}
                                           for bar_chart in range(4)})
        else:
            self.sample_statements.update({"bar_height": {"instances": 5,
                                                          "dist": dist.uniform}})
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
        elif scale == "continuous" or scale == "very_general":
            self.sample_statements.update({"max_height": {"instances": 1,
                                                          "dist": dist.uniform}})
        else:
            assert scale == "fixed", "scale argument given is invalid"

        if random_colour:
            if self.multi_bar_charts:
                colour_samples = {col: {"instances": 4,
                                        "dist": dist.uniform}
                                  for col in ("red", "green", "blue")}
            else:
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
        if multi_bar_charts:
            self.sample_statements.update({"num_bar_charts": {"instances": 1,
                                                              "dist": dist.categorical,
                                                              "output_dim": 4+1},
                                           "legend": {"instances": 1,
                                                      "dist": dist.categorical,
                                                      "output_dim": 2},
                                           "density": {"instances": 1,
                                                       "dist": dist.uniform}
                                           })
        if random_img_dim:
            self.sample_statements.update({"img_width":     {"instances": 1,
                                                             "dist": dist.uniform},
                                           "img_height":    {"instances": 1,
                                                             "dist": dist.uniform}})
        if random_layout:
            self.sample_statements.update({"no_spines": {"instances": 1,
                                                         "dist": dist.categorical,
                                                         "output_dim": 2}})

        self.administrator = Administrator(self.sample_statements,
                                           self.HYPERPARAMS)

        self.obs_embedder = (CNNEmbedder, CNNEmbedder2, CNNEmbedder3)[cnn_number-1](d_emb)

        self.initial_hidden = nn.Parameter(torch.normal(torch.zeros(lstm_layers, 1, hidden_size), 1))
        self.initial_cell = nn.Parameter(torch.normal(torch.zeros(lstm_layers, 1, hidden_size), 1))

        lstm_input_size = d_emb + self.administrator.t_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout)

        self.collect_history = collect_history
        if collect_history is True:
            self.history = []

        if cuda:
            self.cuda()

    def init_lstm(self, obs_embedding):
        """
        run at the start of each trace
        intialises LSTM hidden state etc.
        """
        self.hidden, self.cell = self.initial_hidden, self.initial_cell
        self.instances_dict = {key: -1 for key in self.sample_statements}   # maybe messy to start from -1
        self.obs_embedding = obs_embedding
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

        lstm_input = torch.cat([self.obs_embedding, t], 1).view(1, 1, -1)

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

    def forward(self, observed_image=None, print_params=False):
        x = observed_image.view(1, 3, 210, 210)
        if self.CUDA:
            x = x.cuda()

        if self.collect_history:
            self.history.append({})

        x = self.obs_embedder(x)

        self.init_lstm(x)

        prev_sample_value = None

        if self.wiggle_picture:
            for shift in ("x_shift", "y_shift"):
                ps = self.time_step(shift, prev_sample_value)
                prev_sample_value = pyro.sample(shift,
                                                proposal_dists.categorical_proposal,
                                                ps=ps).type(torch.FloatTensor)

        if self.scale == "discrete":
            max_max_height = 100
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
            max_height = 100
        elif self.scale == "very_general":
            mode, certainty = self.time_step("max_height", prev_sample_value)
            prev_sample_value = pyro.sample("max_log_height",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([-1])),
                                            Variable(torch.Tensor([2.5])),
                                            mode*3.5 - 1,
                                            certainty)
            max_height = 320
        else:
            max_height = 10

        if self.multi_bar_charts:
            ps = self.time_step("num_bar_charts", prev_sample_value)
            num_bar_charts = pyro.sample("num_bar_charts",
                                         proposal_dists.categorical_proposal,
                                         ps=ps)
            prev_sample_value = num_bar_charts.type(torch.FloatTensor)
            num_bar_charts += 1     # to go from 1 to 4
        else:
            num_bar_charts = 1

        if self.random_colour:
            if self.multi_bar_charts:
                for bar_chart in range(num_bar_charts):
                    for colour in ("red", "green", "blue"):
                        mode, certainty = self.time_step(colour,
                                                         prev_sample_value)
                        prev_sample_value = pyro.sample("{}_{}".format(colour, bar_chart),
                                                        proposal_dists.uniform_proposal,
                                                        Variable(torch.Tensor([0])),
                                                        Variable(torch.Tensor([1])),
                                                        mode,
                                                        certainty)
            else:
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
            # mode, certainty = modes[0], certainties[0]
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

        if self.random_img_dim:
            mode, certainty = self.time_step("img_width", prev_sample_value)
            prev_sample_value = pyro.sample("img_width",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([25])),
                                            Variable(torch.Tensor([100])),
                                            mode*75 + 25,
                                            certainty)
            mode, certainty = self.time_step("img_height", prev_sample_value)
            prev_sample_value = pyro.sample("img_height",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([200])),
                                            Variable(torch.Tensor([500])),
                                            mode*300 + 200,
                                            certainty)

        if num_bar_charts > 1:
            mode, certainty = self.time_step("density", prev_sample_value)
            prev_sample_value = pyro.sample("density",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([1])),
                                            mode,
                                            certainty)
            ps = self.time_step("legend", prev_sample_value)
            prev_sample_value = pyro.sample("legend",
                                            proposal_dists.categorical_proposal,
                                            ps=ps).type(torch.FloatTensor)

        if self.random_layout:
            ps = self.time_step("no_spines", prev_sample_value)
            prev_sample_value = pyro.sample("no_spines",
                                            proposal_dists.categorical_proposal,
                                            ps=ps).type(torch.FloatTensor)

        ps = self.time_step("num_bars",
                            prev_sample_value)
        num_bars = pyro.sample("num_bars",
                                proposal_dists.categorical_proposal,
                                ps=ps)
        prev_sample_value = num_bars.type(torch.FloatTensor)

        if self.multi_bar_charts:
            for bar_chart in range(num_bar_charts):
                sample_name = "bar_height_{}".format(bar_chart)
                for _ in range(num_bars):
                    mode, certainty = self.time_step(sample_name,
                                                     prev_sample_value)
                    print(mode.data.numpy()[0])
                    prev_sample_value = pyro.sample("{}_{}".format(sample_name, self.instances_dict[sample_name]),
                                                    proposal_dists.uniform_proposal,
                                                    Variable(torch.Tensor([0])),
                                                    Variable(torch.Tensor([max_height])),
                                                    mode*max_height,
                                                    certainty)
        else:
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
                if print_params:
                    print(mode.data.numpy()[0], '*', max_height, certainty.data.numpy()[0])

    def get_history(self):
        if not self.collect_history:
            raise Exception("collect_history is set to False")
        return self.history
