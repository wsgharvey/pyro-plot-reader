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


class UniformProposalLayer(nn.Module):
    def __init__(self, input_dim):
        super(UniformProposalLayer, self).__init__()
        self.fcn1 = nn.Linear(input_dim, input_dim)
        self.fcn2 = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fcn2(F.relu(self.fcn1(x)))
        modes = x[:, 0]
        certainties = x[:, 1]
        modes = nn.Sigmoid()(modes)
        certainties = nn.Softplus()(certainties)
        return modes, certainties


class QueryLayer(nn.Module):
    def __init__(self, input_dim, n_queries, d_k):
        super(QueryLayer, self).__init__()
        self.n_queries = n_queries
        self.d_k = d_k
        self.fcn1 = nn.Linear(input_dim, n_queries*d_k)
        self.fcn2 = nn.Linear(n_queries*d_k, n_queries*d_k)

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fcn2(F.relu(self.fcn1(x)))
        x = x.view(self.n_queries, self.d_k)
        return x


class Guide(nn.Module):
    def __init__(self, d_k=64, d_v=128, n_queries=16, hidden_size=2048, lstm_layers=1):
        super(Guide, self).__init__()
        self.view_embedder = ViewEmbedder()
        self.location_embedder = LocationEmbeddingMaker(200, 200)
        self.mha = MultiHeadAttention(h=8, d_k=d_k, d_v=d_v, d_model=d_v)
        self.initial_hidden = nn.Parameter(torch.normal(torch.zeros(1, lstm_layers, hidden_size), 1))
        self.initial_cell = nn.Parameter(torch.normal(torch.zeros(1, lstm_layers, hidden_size), 1))
        self.lstm = nn.LSTM(input_size=n_queries*d_v,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=0.1)
        self.proposal_layers = nn.ModuleList([UniformProposalLayer(hidden_size) for _ in range(3)])
        self.query_layers = nn.ModuleList([QueryLayer(hidden_size, n_queries, d_v) for _ in range(3)])

    def forward(self, observed_image=None):
        x = observed_image.view(1, 3, 200, 200)

        pyro.sample("num_bars",
                    dist.categorical,
                    ps=Variable(torch.Tensor(np.array([0., 0., 0., 1., 0., 0.]))))

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

        hidden, cell = self.initial_hidden, self.initial_cell
        for step in range(3):
            queries = self.query_layers[step](hidden)
            lstm_input = self.mha(queries, x, x).view(1, 2048)
            lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

            modes, certainties = self.proposal_layers[step](lstm_output)
            mode, certainty = modes[0], certainties[0]

            if isinstance(mode, torch.cuda.FloatTensor):
                mode = mode.cpu()
                certainty = certainty.cpu()
            print(mode.data.numpy()[0])

            pyro.sample("bar_height_{}".format(step),
                        proposal_dists.uniform_proposal,
                        Variable(torch.Tensor([0])),
                        Variable(torch.Tensor([10])),
                        mode*10,    # TODO: move scaling somewhere else
                        certainty)
