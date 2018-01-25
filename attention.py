import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class LocationEmbeddingMaker(nn.Module):
    def __init__(self, x_range, y_range):
        super(LocationEmbeddingMaker, self).__init__()
        self.x_embedder = nn.Parameter(torch.normal(0, torch.ones(128))/x_range)
        self.y_embedder = nn.Parameter(torch.normal(0, torch.ones(128))/y_range)

    def createLocationEmbedding(self, x, y):
        emb_x = [np.sin(x/(30**(i/64))) for i in range(64)] + \
                [np.cos(x/(30**(i/64))) for i in range(64)]
        emb_y = [np.cos(y/(30**(i/64))) for i in range(64)] + \
                [np.sin(y/(30**(i/64))) for i in range(64)]
        emb = Variable(torch.Tensor(np.array(emb_x) +
                                    np.array(emb_y)))
        if isinstance(self.x_embedder.data, torch.cuda.FloatTensor):
            emb = emb.cuda()
        emb += x*self.x_embedder + y*self.y_embedder
        return emb.view(1, 128)


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
