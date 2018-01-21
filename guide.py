import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.infer.csis.proposal_dists as proposal_dists

import numpy as np


class LocationEmbeddingMaker(nn.Module):
    def __init__(self, x_range, y_range):
        super(LocationEmbeddingMaker, self).__init__()
        # TODO: see if it is good to make these not optimisable
        self.x_embedder = nn.Parameter(torch.normal(0, torch.ones(512))/x_range)
        self.y_embedder = nn.Parameter(torch.normal(0, torch.ones(512))/y_range)

    def createLocationEmbedding(self, x, y):
        emb_x = [np.sin(x/(30**(i/256))) for i in range(256)] + \
                [np.cos(x/(30**(i/256))) for i in range(256)]
        emb_y = [np.cos(y/(30**(i/256))) for i in range(256)] + \
                [np.sin(y/(30**(i/256))) for i in range(256)]
        emb = Variable(torch.Tensor(np.array(emb_x) +
                                    np.array(emb_y)))
        if isinstance(self.x_embedder.data, torch.cuda.FloatTensor):
            emb = emb.cuda()
        emb += x*self.x_embedder + y*self.y_embedder
        return emb.view(1, 512)


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
        self.location_embedder = LocationEmbeddingMaker(200, 200)
        self.view_embedder = LocalEmbedder()

    def forward(self, x):
        x = x.view(1, 3, 200, 200)

        # find and embed each seperate locations
        locations = (x[:, :, 10*j:10*(j+2), 10*i:10*(i+2)].clone() for i in range(19) for j in range(19))
        local_embeddings = [self.view_embedder(loc) for loc in locations]
        x = torch.cat(local_embeddings, 0)

        # add location embeddings
        if isinstance(x.data, torch.cuda.FloatTensor):
            location_embeddings = [self.location_embedder.createLocationEmbedding(i, j).cuda()
                                   for i in range(0, 190, 10)
                                   for j in range(0, 190, 10)]
        else:
            location_embeddings = [self.location_embedder.createLocationEmbedding(i, j)
                                   for i in range(0, 190, 10)
                                   for j in range(0, 190, 10)]

        x = x + torch.cat(location_embeddings, 0)

        # run everything
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention = MultiHeadAttention()
        self.query = nn.Parameter((torch.rand(20, 512)))
        self.fcn = nn.Linear(512*20, 16)

    def forward(self, x):
        x = F.relu(self.attention(self.query, x, x))
        x = x.view(512*20)
        return self.fcn(x)


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.encoder = Encoder(N=1)
        self.decoder = Decoder()

    def forward(self, observed_image=None):
        assert observed_image is not None
        observed_image = F.relu(torch.floor(observed_image - F.relu(observed_image-255)))    # simulate getting turned into a png and back
        img = observed_image.view(1, 3, 200, 200)

        encoding = self.encoder(img)
        decoded = self.decoder(encoding)

        bar_heights = nn.Sigmoid()(decoded[:5])*10
        certainties = nn.Softplus()(decoded[5:10])
        n_bars_probs = nn.Softmax()(decoded[10:16].view(1, -1)).view(-1)

        if isinstance(bar_heights.data, torch.cuda.FloatTensor):
            bar_heights = bar_heights.cpu()
            certainties = certainties.cpu()
            n_bars_probs = n_bars_probs.cpu()

        num_bars = pyro.sample("num_bars",
                               dist.categorical,
                               ps=n_bars_probs)
        num_bars = num_bars.data.numpy()[0]

        for bar_num in range(num_bars):
            mode = bar_heights[bar_num]
            try:
                print(mode.data.numpy()[0])
            except:
                print(mode.data.cpu().numpy()[0])
            pyro.sample("bar_height_{}".format(bar_num),
                        proposal_dists.uniform_proposal,
                        Variable(torch.Tensor([0])),
                        Variable(torch.Tensor([10])),
                        mode,
                        certainties[bar_num])
