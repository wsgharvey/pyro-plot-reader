import pyro
import pyro.infer
import pyro.distributions as dist

import torch
from torch.autograd import Variable

import numpy as np

from helpers import fig2tensor,\
                    set_size_pixels

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def model(observed_image=Variable(torch.zeros(200, 200))):
    """ generates a bar chart with a single bar
    """
    max_height = 10
    height, width = 200, 200

    num_bars = 3

    bar_heights = []
    for bar_num in range(3):
        bar_height = pyro.sample("bar_height_{}".format(bar_num),
                                 dist.uniform,
                                 Variable(torch.Tensor([0])),
                                 Variable(torch.Tensor([max_height])))
        bar_heights.append(bar_height.data.numpy()[0])

    # colour = pyro.sample("colour".format(bar_num),
    #                      dist.categorical,
    #                      ps=Variable(torch.ones(7))/7,
    #                      vs=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    fig, ax = plt.subplots()
    ax.bar(range(num_bars),
           bar_heights,
           label="Bar")
    ax.set_ylim(0, max_height)

    # get the graph as a matrix
    fig = set_size_pixels(fig, (width, height))
    image = Variable(fig2tensor(fig))
    plt.close()

    flattened_image = image.view(-1)
    noise_std = Variable(torch.ones(flattened_image.size()))
    flattened_obs_image = observed_image.view(-1)
    observed_image = pyro.observe("observed_image",
                                  dist.normal,
                                  obs=flattened_obs_image,
                                  mu=flattened_image,
                                  sigma=noise_std)
    return {"image": image,
            "bar_heights": np.array(bar_heights)}
