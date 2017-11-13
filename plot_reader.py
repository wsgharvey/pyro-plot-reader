import torch
import torch.nn as nn
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

from helpers import fig2tensor,\
                    set_size_pixels
from nn import BarHeightNet


class PlotReader(nn.Module):
    def __init__(self):
        super(PlotReader, self).__init__()
        self.height = 200
        self.width = 200
        self.bar_height_net = BarHeightNet()
        self.abc_dist_size = 50
        self.abc_projection = Variable(torch.randn(self.abc_dist_size,
                                                   3*self.height*self.width)) / 1000

    def model(self, observed_image,
              return_image=False,
              save_address=None):
        """ generates a bar chart with a single bar
        """
        max_height = 10
        height, width = self.height, self.width

        bar_height = pyro.sample("bar_height",
                                 dist.uniform,
                                 Variable(torch.Tensor([0])),
                                 Variable(torch.Tensor([max_height])))

        # plot the graph
        fig, ax = plt.subplots()
        ax.bar([1],
               bar_height.data,
               label="Bar")
        ax.set_ylim(0, max_height)

        # get the graph as a matrix
        fig = set_size_pixels(fig, (width, height))
        if save_address is not None:
            fig.savefig(save_address)
        image = Variable(fig2tensor(fig))
        plt.close()

        if observed_image is not None:
            observed_image = torch.mv(self.abc_projection,
                                      observed_image.view(-1))
            generated_image = torch.mv(self.abc_projection,
                                       image.view(-1))
            noise_std = Variable(torch.ones(generated_image.size()))
            observed_image = pyro.observe("observed_image",
                                          dist.normal,
                                          observed_image,
                                          generated_image,
                                          noise_std)
        # must return values in a structure that allows equality comparison
        # when running inference
        if return_image:
            return bar_height, image
        else:
            return bar_height

    def guide(self, observed_image):
        pyro.module("bar_height_net", self.bar_height_net)
        mean_height = self.bar_height_net(observed_image)
        bar_height = pyro.sample("bar_height",
                                 dist.normal,
                                 mean_height,
                                 Variable(torch.ones(mean_height.size())))
        print("height sampled as", bar_height.data.numpy()[0])
