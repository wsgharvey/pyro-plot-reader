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

    def model(self,
              observed_image=Variable(torch.zeros(200, 200))):
        """ generates a bar chart with a single bar
        """
        max_height = 10
        height, width = self.height, self.width

        bar_height = pyro.sample("bar_height",
                                 dist.uniform,
                                 Variable(torch.Tensor([0])),
                                 Variable(torch.Tensor([max_height])))
        fig, ax = plt.subplots()
        ax.bar([1],
               bar_height.data,
               label="Bar")
        ax.set_ylim(0, max_height)

        # get the graph as a matrix
        fig = set_size_pixels(fig, (width, height))
        image = Variable(fig2tensor(fig))
        plt.close()

        generated_image = image.view(-1)
        noise_std = Variable(10*torch.ones(generated_image.size()))
        observed_image = pyro.observe("observed_image",
                                      dist.normal,
                                      obs=observed_image,
                                      mu=generated_image,
                                      sigma=noise_std)

        return generated_image
