import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist

import numpy as np

global d_model
d_model = 512


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
    def __init__(self, h=4, d_k=128, d_v=256):
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
    def __init__(self, dff=2048):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention()
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


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.encoder = Encoder(N=4)
        self.decoder = nn.Linear(51200, 3)

        self.log_std = nn.Parameter(torch.Tensor([1]))

    def forward(self, observed_image=None):
        assert observed_image is not None
        observed_image = F.relu(torch.floor(observed_image - F.relu(observed_image-255)))    # simulate getting turned into a png and back
        img = observed_image.view(1, 3, 200, 200)

        encoding = self.encoder(img)
        bar_heights = self.decoder(encoding.view(51200)).view(3)
        std = self.log_std.exp()

        for bar_num in range(3):
            mean = bar_heights[bar_num]
            print(mean.data.numpy()[0])
            pyro.sample("bar_height_{}".format(bar_num),
                        dist.normal,
                        mean,
                        std)


# class AttentionBox(nn.Module):
#     """
#     takes in low res image, returns attention weight matrix
#     """
#     def __init__(self):
#         super(AttentionBox, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
#         self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.conv3 = nn.Conv2d(10, 5, 3, padding=1)
#         self.fcn1 = nn.Linear(500, 50)
#         self.fcn2 = nn.Linear(50, 10)
#         self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         x = x.view(1, 3, 20, 20)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = x.view(1, 500)
#         x = F.relu(self.fcn1(x))
#         x = self.fcn2(x)
#         x = nn.Softmax()(x)
#         x = x.view(1, 10)
#         return x
#
#
# class FocusBox(nn.Module):
#     def __init__(self):
#         super(FocusBox, self).__init__()
#         self.conv1 = nn.Conv2d(40, 40, 3)
#
#     def forward(self, x):
#         """
#         x is 1x40x200x20
#         """
#         x = x.view(1, 40, 200, 20)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv4(x))
#         x = self.pool3(x)
#         x = F.relu(self.conv5(x))
#
#         x = x.view(1, 20*23)
#         x = F.relu(self.fcn1(x))
#         x = self.fcn2(x)
#         return x.view(1)
#
#
# class Guide(nn.Module):
#     def __init__(self):
#         super(Guide, self).__init__()
#         # for calculating attention weights
#         self.pool = nn.AvgPool2d(10, stride=10)
#         # self.attention_boxes = [AttentionBox() for _ in range(3)]   # todo with this innit
#         # self.focus_boxes = [FocusBox() for _ in range(10)]
#         for i in range(3):
#             exec("self.attention_box_{} = AttentionBox()".format(i))
#         for i in range(10):
#             exec("self.focus_box_{} = FocusBox()".format(i))
#
#         self.full_conv1 = nn.Conv2d(3, 20, 3, padding=1)
#         self.full_conv2 = nn.Conv2d(20, 40, 3, padding=1)
#         self.full_conv3 = nn.Conv2d(40, 40, 3, padding=1)
#         self.full_conv4 = nn.Conv2d(40, 40, 3, padding=1)
#
#         self.log_std = nn.Parameter(torch.Tensor([1]))
#
#     def forward(self, observed_image=None):
#         assert observed_image is not None
#         observed_image = F.relu(torch.floor(observed_image - F.relu(observed_image-255)))    # simulate getting turned into a png and back
#         img = observed_image.view(1, 3, 200, 200)
#         low_res_img = self.pool(img)
#
#         img_emb = F.relu(self.full_conv1(img))
#         img_emb = F.relu(self.full_conv2(img_emb))
#         img_emb = F.relu(self.full_conv3(img_emb))
#         img_emb = self.full_conv4(img_emb)
#
#         std = self.log_std.exp()
#
#         predictions = Variable(torch.Tensor(10, 1))
#         for i in range(10):
#             start, end = i*20, (i+1)*20
#             img_slice = img_emb[:, :, :, start:end]
#             img_slice.contiguous()
#             exec("global slice_pred; slice_pred = self.focus_box_{}(img_slice)".format(i))
#             predictions[i] = slice_pred
#
#         for bar_num in range(3):
#             exec("global attention_weights; attention_weights = self.attention_box_{}(low_res_img)".format(bar_num))
#             # now dot the weights with the embedding
#             # the troubles are happening
#             mean = sum(w*x for w, x in zip(attention_weights.view(-1), predictions))
#             print(mean.data.numpy()[0])
#             pyro.sample("bar_height_{}".format(bar_num),
#                         dist.normal,
#                         mean,
#                         std)
