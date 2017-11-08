import torch
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from PIL import Image

from helpers import fig2tensor,\
                    set_size_pixels


def graph(return_image=False):
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
    plt.close()

    # observe the graph with some Gaussian noise
    noise_std = Variable(torch.ones(image.size()))
    observed_image = pyro.sample("observed_image",
                                 dist.normal,
                                 image.view(-1),
                                 noise_std.view(-1))
    # must return values in a structure that allows equality comparison
    if return_image:
        return bar_height, image
    else:
        return bar_height


# make data:
real_height, real_img = graph(return_image=True)
print("true height is", real_height)

# condition on the data
conditioned_graph = pyro.condition(
                        graph,
                        data={"observed_image": real_img.view(-1)})

# run inference
posterior = pyro.infer.Importance(conditioned_graph, num_samples=15)
marginal = pyro.infer.Marginal(posterior)

# sample from empirical posterior
print("estimated height is:\n", marginal())
