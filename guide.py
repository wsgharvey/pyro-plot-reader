import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np


def createLocationEmbedding(x, y):
    emb = [np.sin(x/(10000**(2*i/256))) for i in range(256)] + \
          [np.cos(x/(10000**(2*i/256))) for i in range(256)]
    return Variable(torch.Tensor(np.array(emb))).view(1, 512)


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        weights = torch.mm(Q, K.transpose(0, 1))    # K transpose?
        d_k = K.size()[0]
        weights /= d_k**0.5
        weights = torch.nn.Softmax()(weights)

        return torch.mm(weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, h=1, d_k=128, d_v=256, d_model=512):
        """
        :h: number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.fcn_qs = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(h)])
        self.fcn_ks = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(h)])
        self.fcn_vs = nn.ModuleList([nn.Linear(d_model, d_v, bias=False) for _ in range(h)])
        self.fcn_out = nn.Linear(h*d_v, d_model, bias=False)

        self.dpa = DotProductAttention()

    def forward(self, Q, K, V):
        """
        :Q: matrix of queries - n_locations x d_model
        :K: matrix of queries - n_locations x d_model
        :V: matrix of queries - n_locations x d_model

        :returns: n_locations x d_model
        """
        head_outputs = []
        for i in range(self.h):
            Q_emb = self.fcn_qs[i](Q)
            K_emb = self.fcn_ks[i](K)
            V_emb = self.fcn_vs[i](V)
            head_outputs.append(self.dpa(Q_emb, K_emb, V_emb))
        x = torch.cat(head_outputs, 1)
        x = self.fcn_out(x)
        return x


class LocalEmbedder(nn.Module):
    def __init__(self):
        super(LocalEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.conv7 = nn.Conv2d(64, 128, 3)
        self.conv8 = nn.Conv2d(128, 128, 3)
        self.conv9 = nn.Conv2d(128, 256, 3)
        self.conv10 = nn.Conv2d(256, 512, 2)

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
        x = x.view(1, 512)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, dff=2048):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(h=1)
        self.fcn1 = nn.Linear(d_model, dff)
        self.fcn2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = x + self.attention(x, x, x)
        x = x + self.fcn2(F.relu(self.fcn1(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N)])
        self.local_embedder = LocalEmbedder()

    def forward(self, x):
        x = x.view(1, 3, 200, 200)

        # find and embed each seperate locations
        locations = (x[:, :, 20*j:20*(j+1), 20*i:20*(i+1)].clone() for i in range(10) for j in range(10))
        local_embeddings = [self.local_embedder(loc) for loc in locations]
        x = torch.cat(local_embeddings, 0)

        # add location embeddings
        location_embeddings = [createLocationEmbedding(i, j)
                               for i in range(0, 200, 20)
                               for j in range(0, 200, 20)]
        x = x + torch.cat(location_embeddings, 0)

        # run everything
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention = MultiHeadAttention()
        self.query = nn.Parameter(Variable(torch.rand(1, 512)))
        self.fcn = nn.Linear(512, 6)

    def forward(self, x):
        x = F.relu(self.attention(self.query, x, x))
        return self.fcn(x)


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.encoder = Encoder(N=2)
        self.decoder = nn.Linear(51200, 6)

    def forward(self, observed_image=None):
        assert observed_image is not None
        observed_image = F.relu(torch.floor(observed_image - F.relu(observed_image-255)))    # simulate getting turned into a png and back
        img = observed_image.view(1, 3, 200, 200)

        encoding = self.encoder(img)
        decoded = self.decoder(encoding.view(51200))

        bar_heights = nn.Sigmoid()(decoded[:3])*10
        certainties = nn.Softplus()(decoded[3:])

        for bar_num in range(3):
            mean = bar_heights[bar_num]
            print(mean.data.numpy()[0])
            pyro.sample("bar_height_{}".format(bar_num),
                        proposal_dists.uniform_proposal,
                        Variable(torch.Tensor([0])),
                        Variable(torch.Tensor([10])),
                        mean,
                        certainties[bar_num])
