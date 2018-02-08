import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class LocationEmbeddingMaker(nn.Module):
    def __init__(self, x_range, y_range, add_linear_embedding=True):
        """
        x_range is the number ox pixels in the x-direction
        y_range is the number of pixels in the y-direction
        """
        super(LocationEmbeddingMaker, self).__init__()
        self.add_linear_embedding = add_linear_embedding
        self.x_embedder = nn.Parameter(torch.normal(0, torch.ones(128))/x_range)
        self.y_embedder = nn.Parameter(torch.normal(0, torch.ones(128))/y_range)

    def createLocationEmbedding(self, x, y, x_offset=0, y_offset=0):
        """
        x, y are the coordinates of the location being embedded
        x_offset, y_offset allow an offset to simulate the image being translated
        """
        x = x + x_offset
        y = y + y_offset
        emb_x = [np.sin(x/(30**(i/64))) for i in range(64)] + \
                [np.cos(x/(30**(i/64))) for i in range(64)]
        emb_y = [np.cos(y/(30**(i/64))) for i in range(64)] + \
                [np.sin(y/(30**(i/64))) for i in range(64)]
        emb = Variable(torch.Tensor(np.array(emb_x) +
                                    np.array(emb_y)))
        if isinstance(self.x_embedder.data, torch.cuda.FloatTensor):
            emb = emb.cuda()
        if self.add_linear_embedding:
            emb += x*self.x_embedder + y*self.y_embedder
        return emb.view(1, 128)


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, Q, K, V, return_graphic=False):
        weights = torch.mm(Q, K.transpose(0, 1))    # K transpose?
        d_k = K.size()[0]
        weights /= d_k**0.5
        weights = torch.nn.Softmax()(weights)
        result = torch.mm(weights, V)

        if not return_graphic:
            return result
        else:
            graphic = np.zeros((21, 21))
            weights = weights.data
            if isinstance(weights, torch.cuda.FloatTensor):
                weights = weights.cpu()
            weights = weights.numpy()
            for query in weights:
                for i in range(19):
                    for j in range(19):
                        graphic[j*1:j*1+2, i*1:i*1+2] += query[i*19+j]
            graphic = np.repeat(graphic, 10, axis=0)
            graphic = np.repeat(graphic, 10, axis=1)
            return result, graphic


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

    def forward(self, Q, K, V, attention_tracker=None):
        """
        :Q: matrix of queries - n_locations x d_model
        :K: matrix of queries - n_locations x d_model
        :V: matrix of queries - n_locations x d_model

        :returns: n_locations x d_model
        """
        if attention_tracker is not None:
            graphic = np.zeros((200, 200))

        head_outputs = []
        for i in range(self.h):
            Q_emb = self.fcn_qs[i](Q)
            K_emb = self.fcn_ks[i](K)
            V_emb = self.fcn_vs[i](V)
            if attention_tracker is None:
                head_outputs.append(self.dpa(Q_emb, K_emb, V_emb))
            else:
                head_output, head_locations = self.dpa(Q_emb, K_emb, V_emb,
                                                       return_graphic=True)
                head_outputs.append(head_output)
                graphic += head_locations

        x = torch.cat(head_outputs, 1)
        x = self.fcn_out(x)

        if attention_tracker is not None:
            graphic = graphic * 255 / np.amax(graphic)
            attention_tracker.add_graphic(graphic)

        return x
