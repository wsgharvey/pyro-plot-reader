import pyro
import pyro.infer
import pyro.distributions as dist

import torch
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt

from helpers import fig2tensor,\
                    set_size_pixels


def model(observed_image=Variable(torch.zeros(200, 200))):
    """ generates a bar chart with a single bar
    """
    max_height = 10
    height, width = 200, 200

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

    flattened_image = image.view(-1)
    noise_std = Variable(10*torch.ones(flattened_image.size()))
    flattened_obs_image = observed_image.view(-1)
    observed_image = pyro.observe("observed_image",
                                  dist.normal,
                                  obs=flattened_obs_image,
                                  mu=flattened_image,
                                  sigma=noise_std)

    return {"image": image,
            "bar_height": bar_height}
