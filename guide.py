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
import act
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
                 share_smp_embedder=False,
                 share_qry_layer=False,
                 share_prop_layer=False,
                 keys_use_view=True,
                 keys_use_loc=True,
                 vals_use_view=True,
                 vals_use_loc=True,
                 learn_loc_embs=False,
                 wiggle_picture=False,
                 max_loc_emb_wiggle=0,
                 add_linear_loc_emb=True,
                 ponder_cost=1e-3,
                 random_colour=True,
                 random_bar_width=True,
                 random_line_colour=True,
                 random_line_width=True,
                 scale="fixed",
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
                            "low_res_view_as_attention_loc": low_res_view_as_attention_loc,
                            "CUDA": cuda,
                            "share_smp_embedder": share_smp_embedder,
                            "share_qry_layer": share_qry_layer,
                            "share_prop_layer": share_prop_layer,
                            "keys_use_view": keys_use_view,
                            "keys_use_loc": keys_use_loc,
                            "vals_use_view": vals_use_view,
                            "vals_use_loc": vals_use_loc,
                            "max_loc_emb_wiggle": max_loc_emb_wiggle,
                            "ponder_cost": ponder_cost}
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

        self.halting_unit = act.HaltingUnit(hidden_size)

        lstm_input_size = n_queries*d_emb + self.administrator.t_dim + 1    # +1 for ACT bit
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
        self.added_loss = Variable(torch.Tensor([0]))
        if self.CUDA:
            self.added_loss = self.added_loss.cuda()

    def program_step(self, current_sample_name, prev_sample_value):
        """
        perform one LSTM time step
        returns proposal parameters for `current_sample_name`
        """
        self.current_sample_name = current_sample_name
        self.prev_sample_value = prev_sample_value

        self.instances_dict[current_sample_name] += 1
        self.current_instance = self.instances_dict[current_sample_name]

        t = self.administrator.t(self.current_instance,
                                 current_sample_name,
                                 self.prev_instance,
                                 self.prev_sample_name,
                                 prev_sample_value,
                                 self.low_res_emb)

        act_output = self.ACT_step(t)

        proposal_params = self.administrator.get_proposal_layer(current_sample_name, self.current_instance)(act_output)

        self.prev_sample_name = current_sample_name
        self.prev_instance = self.current_instance

        if self.CUDA:
            try:
                proposal_params = proposal_params.cpu()
            except AttributeError:
                proposal_params = tuple(param.cpu() for param in proposal_params)

        if self.collect_history:
            self.history[-1]["{}_{}".format(current_sample_name, self.current_instance)] = proposal_params

        return proposal_params

    def init_ACT(self):
        """
        should this exist?
        """
        self.eps = Variable(torch.Tensor([0.05]))
        if self.cuda:
            self.eps = self.eps.cuda()
        self.remainders = 0

    def ACT_step(self, t):
        halting_weight_sum = Variable(torch.Tensor([0]))
        num_steps = 0
        output = 0

        first_computation_marker = Variable(torch.Tensor([[1]]))
        last_computation_step = False
        if self.cuda:
            first_computation_marker = first_computation_marker.cuda()
            halting_weight_sum = halting_weight_sum.cuda()
        while not last_computation_step:
            num_steps += 1
            lstm_output = self.lstm_step(t, first_computation_marker)

            halting_weight = self.halting_unit(lstm_output)
            if halting_weight_sum + halting_weight > 1-self.eps or num_steps >= 10:
                halting_weight = remainder = 1-halting_weight_sum
                last_computation_step = True

            output += halting_weight*lstm_output
            halting_weight_sum += halting_weight

            first_computation_marker *= 0
        self.remainders += remainder
        print(self.current_sample_name, num_steps)
        return output

    def lstm_step(self, t, first_computation_marker):
        K = self.keys
        V = self.values
        query_layer = self.administrator.get_query_layer(self.current_sample_name, self.current_instance)
        t = torch.cat([t, first_computation_marker], 1)
        queries = query_layer(t=t, prev_hidden=self.hidden)

        if self.attention_tracker is None:
            attention_output = self.mha(queries, K, V).view(1, -1)
        else:
            attention_output = self.mha(queries, K, V, self.attention_tracker).view(1, -1)

        lstm_input = torch.cat([attention_output, t], 1).view(1, 1, -1)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (self.hidden, self.cell))
        del self.hidden
        del self.cell
        self.hidden, self.cell = hidden, cell
        return lstm_output

    def forward(self, observed_image=None):
        x = observed_image.view(1, 3, 210, 210)
        if self.CUDA:
            x = x.cuda()

        if self.collect_history:
            self.history.append({})

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

        if self.HYPERPARAMS["use_low_res_view"] and self.HYPERPARAMS["low_res_view_as_attention_loc"]:
            keys = torch.cat([keys, low_res_emb.view(1, -1)], 0)
            values = torch.cat([values, low_res_emb.view(1, -1)], 0)
        # low_res_img = nn.AvgPool2d(10)(observed_image.view(1, 3, 200, 200))
        # full_pic_embedding = self.big_picture_embedder(low_res_img)
        # x = torch.cat([view_embeddings, full_pic_embedding], 0)
        """"""

        if self.HYPERPARAMS["use_low_res_view"] and not self.HYPERPARAMS["low_res_view_as_attention_loc"]:
            self.init_lstm(keys, values, low_res_emb)
        else:
            self.init_lstm(keys, values, None)
        self.init_ACT()
        prev_sample_value = None

        if self.wiggle_picture:
            for shift in ("x_shift", "y_shift"):
                ps = self.program_step(shift, prev_sample_value)
                prev_sample_value = pyro.sample(shift,
                                                proposal_dists.categorical_proposal,
                                                ps=ps).type(torch.FloatTensor)

        if self.scale == "discrete":
            ps = self.time_step("max_height", prev_sample_value)
            prev_sample_value = pyro.sample("max_height",
                                            proposal_dists.categorical_proposal,
                                            ps=ps).type(torch.FloatTensor)
            max_height = prev_sample_value
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

        if self.random_colour:
            for colour in ("red", "green", "blue"):
                mode, certainty = self.program_step(colour,
                                                    prev_sample_value)
                # mode, certainty = modes[0], certainties[0]
                prev_sample_value = pyro.sample(colour,
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_bar_width:
            mode, certainty = self.program_step("bar_width",
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
                mode, certainty = self.program_step("line_{}".format(colour),
                                                    prev_sample_value)
                # mode, certainty = modes[0], certainties[0]
                prev_sample_value = pyro.sample("line_{}".format(colour),
                                                proposal_dists.uniform_proposal,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1])),
                                                mode,
                                                certainty)
        if self.random_line_width:
            mode, certainty = self.program_step("line_width",
                                                prev_sample_value)
            # mode, certainty = modes[0], certainties[0]
            prev_sample_value = pyro.sample("line_width",
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            Variable(torch.Tensor([2.5])),
                                            mode*2.5,
                                            certainty)

        ps = self.program_step("num_bars",
                               prev_sample_value)
        num_bars = pyro.sample("num_bars",
                                proposal_dists.categorical_proposal,
                                ps=ps)
        prev_sample_value = num_bars.type(torch.FloatTensor)

        for _ in range(num_bars):
            mode, certainty = self.program_step("bar_height",
                                                prev_sample_value)
            # mode, certainty = modes[0], certainties[0]
            prev_sample_value = pyro.sample("{}_{}".format("bar_height", self.instances_dict["bar_height"]),
                                            proposal_dists.uniform_proposal,
                                            Variable(torch.Tensor([0])),
                                            max_height,
                                            mode*max_height.item(),
                                            certainty)

        # a hack to add a term to the loss to limit computation time
        # TODO: must not happen at inference time or will fuck up weights
        if self.cuda:
            self.remainders = self.remainders.cpu()
        pyro.sample("N/A - Adding Loss Term",   # could this be changed to observe 0? maybe that would stop it showing warnings
                    dist.uniform,
                    Variable(torch.Tensor([0])),
                    (self.HYPERPARAMS["ponder_cost"]*self.remainders).exp())

        if self.attention_tracker is not None:
            self.attention_tracker.save_graphics()

    def get_history(self):
        if not self.collect_history:
            raise Exception("collect_history is set to False")
        return self.history
