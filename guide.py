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
    """
    embeds a 3x20x20 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(ViewEmbedder, self).__init__()
        self.output_dim = output_dim                        # 20x20
        self.conv1 = nn.Conv2d(3, 8, 3)                     # 18x18
        self.conv2 = nn.Conv2d(8, 8, 3)                     # 16x16
        self.conv3 = nn.Conv2d(8, 16, 3)                    # 14x14
        self.conv4 = nn.Conv2d(16, 16, 3)                   # 12x12
        self.conv5 = nn.Conv2d(16, 16, 3)                   # 10x10
        self.conv6 = nn.Conv2d(16, 32, 3)                   # 8x8
        self.conv7 = nn.Conv2d(32, 64, 3)                   # 6x6
        self.conv8 = nn.Conv2d(64, 64, 3)                   # 4x4
        self.conv9 = nn.Conv2d(64, max(64, output_dim), 3)  # 2x2
        self.conv10 = nn.Conv2d(256, output_dim, 2)         # 1x1

    def forward(self, x):
        x = x.view(1, 3, 20, 20)
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        x = x.view(1, self.output_dim)
        return x

class FullViewEmbedder(nn.Module):
    """
    embeds a 3x20x20 pixel region into a vector of size `output_dim`
    """
    def __init__(self, output_dim):
        super(ViewEmbedder, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.conv6 = nn.Conv2d(16, 32, 3)
        self.conv7 = nn.Conv2d(32, 64, 3)
        self.conv8 = nn.Conv2d(64, 64, 3)
        self.conv9 = nn.Conv2d(64, output_dim, 3)

    def forward(self, x):
        x = x.view(1, 3, 21, 21)
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        x = x.view(1, self.output_dim)
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
                 d_emb=128,
                 n_queries=16,
                 hidden_size=2048,
                 lstm_layers=1,
                 smp_emb_dim=32,
                 n_attention_queries=20,
                 n_attention_heads=8,
                 lstm_dropout=0.1,
                 use_overall_view=True,
                 low_res_emb_size=64,
                 cuda=False,
                 share_smp_embedder=True,
                 share_qry_layer=True,
                 share_prop_layer=True,
                 keys_use_view=True,
                 keys_use_loc=True,
                 vals_use_view=True,
                 vals_use_loc=True,
                 wiggle_picture=True,
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
                            "d_emb": d_emb,
                            "n_queries": n_queries,
                            "hidden_size": hidden_size,
                            "lstm_layers": lstm_layers,
                            "smp_emb_dim": smp_emb_dim,
                            "n_attention_queries": n_attention_queries,
                            "n_attention_heads": n_attention_heads,
                            "lstm_dropout": lstm_dropout,
                            "use_overall_view": use_overall_view,
                            "low_res_emb_size": low_res_emb_size,
                            "CUDA": cuda,
                            "share_smp_embedder": share_smp_embedder,
                            "share_qry_layer": share_qry_layer,
                            "share_prop_layer": share_prop_layer,
                            "keys_use_view": keys_use_view,
                            "keys_use_loc": keys_use_loc,
                            "vals_use_view": vals_use_view,
                            "vals_use_loc": vals_use_loc,
                            "wiggle_picture": wiggle_picture,
                            "max_loc_emb_wiggle": max_loc_emb_wiggle}
        self.CUDA = cuda
        self.random_colour = random_colour
        self.random_bar_width = random_bar_width
        self.random_line_colour = random_line_colour
        self.random_line_width = random_line_width

        self.sample_statements = {"num_bars":   {"instances": 1,
                                                 "dist": dist.categorical,
                                                 "output_dim": 5},
                                  "bar_height": {"instances": 5,
                                                 "dist": dist.uniform}}
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
        if use_overall_view:
            self.low_res_embedder = FullViewEmbedder(output_dim=low_res_emb_size)
        self.location_embedder = LocationEmbeddingMaker(200, 200, add_linear_loc_emb)
        self.mha = MultiHeadAttention(h=n_attention_heads, d_k=d_k, d_v=d_emb, d_model=d_emb)
        self.initial_hidden = nn.Parameter(torch.normal(torch.zeros(1, lstm_layers, hidden_size), 1))
        self.initial_cell = nn.Parameter(torch.normal(torch.zeros(1, lstm_layers, hidden_size), 1))

        lstm_input_size = n_queries*d_emb + self.administrator.t_dim
        if use_overall_view:
            lstm_input_size += low_res_emb_size
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

    def init_lstm(self, input_embeddings, low_res_emb=None):
        """
        run at the start of each trace
        intialises LSTM hidden state etc.
        """
        self.hidden, self.cell = self.initial_hidden, self.initial_cell
        self.instances_dict = {key: -1 for key in self.sample_statements}   # maybe messy to start from -1
        self.x = input_embeddings
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
                                 prev_sample_value)
        queries = self.administrator.get_query_layer(current_sample_name, current_instance)(self.hidden, t)   # this should use sample_name not step
        if self.attention_tracker is None:
            attention_output = self.mha(queries, self.x, self.x).view(1, -1)
        else:
            attention_output = self.mha(queries, self.x, self.x, self.attention_tracker).view(1, -1)

        if self.low_res_emb is None:
            lstm_input = torch.cat([attention_output, t], 1).view(1, 1, -1)
        else:
            lstm_input = torch.cat([attention_output, lor_res_emb, t], 1).view(1, 1, -1)

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

        if self.use_overall_view:
            low_res_img = nn.AvgPool(10)(x)
            low_res_emb = self.low_res_embedder(low_res_img)

        """ this bit should probably be moved """
        # find and embed each seperate location
        views = (x[:, :, 10*j:10*(j+2), 10*i:10*(i+2)].clone() for i in range(20) for j in range(20))
        view_embeddings = [self.view_embedder(view) for view in views]

        # add location embeddings
        x_offset = np.random.uniform(0, self.HYPERPARAMS["max_loc_emb_wiggle"])  # will be 0 by default
        y_offset = np.random.uniform(0, self.HYPERPARAMS["max_loc_emb_wiggle"])  # will be 0 by default
        location_embeddings = [self.location_embedder.createLocationEmbedding(i, j, x_offset=x_offset, y_offset=y_offset)
                               for i in range(0, 200, 10)
                               for j in range(0, 200, 10)]
        if isinstance(x.data, torch.cuda.FloatTensor):
            location_embeddings = [emb.cuda() for emb in location_embeddings]

        view_embeddings = torch.cat(view_embeddings, 0)
        location_embeddings = torch.cat(location_embeddings, 0)

        keys = Variable(torch.zeros(view_embeddings.shape))
        values = Variable(torch.zeros(view_embeddings.shape))
        if self.HYPERPARAMS["keys_use_view"]:
            keys += view_embeddings
        if self.HYPERPARAMS["keys_use_loc"]:
            keys += location_embeddings
        if self.HYPERPARAMS["vals_use_view"]:
            values += view_embeddings
        if self.HYPERPARAMS["vals_use_loc"]:
            values += location_embeddings
        # low_res_img = nn.AvgPool2d(10)(observed_image.view(1, 3, 200, 200))
        # full_pic_embedding = self.big_picture_embedder(low_res_img)
        # x = torch.cat([view_embeddings, full_pic_embedding], 0)
        """"""

        if self.use_overall_view:
            self.init_lstm(x, low_res_emb)
        else:
            self.init_lstm(x, None)
        prev_sample_value = None

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
            prev_sample_value = pyro.sample(colour,
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
                prev_sample_value = pyro.sample(colour,
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_line_width:
            modes, certainties = self.time_step("line_width",
                                                prev_sample_value)
            mode, certainty = modes[0], certainties[0]
            prev_sample_value = pyro.sample(colour,
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([1])),
                                            mode,
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
            mode, certainty = modes[0]*10, certainties[0]
            print(mode.data.numpy()[0])
            prev_sample_value = pyro.sample("{}_{}".format("bar_height", self.instances_dict["bar_height"]),
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
