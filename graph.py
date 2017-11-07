import torch
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from helpers import fig2tensor,\
                    set_size_pixels


def graph():
    """ generates a bar chart with a single bar """
    max_height = 10
    height, width = 200, 200

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
    image = Variable(fig2tensor(fig))

    # observe the graph with some Gaussian noise
    noise_std = Variable(torch.ones(image.size()))
    observed_image = pyro.sample("observed_image",
                                 dist.normal,
                                 image,
                                 noise_std)
    return bar_height


print(graph())
