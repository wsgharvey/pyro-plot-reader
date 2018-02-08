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


def model(observed_image=Variable(torch.zeros(200, 200)),
          random_colour=True,
          random_bar_width=True,
          random_line_colour=True,
          random_line_width=True,
          wiggle_picture=True):
    max_height = 10
    max_line_width = 2.5
    max_translation = 10
    height, width = 200, 200

    if wiggle_picture:
        x_shift = pyro.sample("x_shift",
                              dist.categorical,
                              ps=Variable(torch.ones(max_translation))).data.numpy()[0]

        y_shift = pyro.sample("y_shift",
                              dist.categorical,
                              ps=Variable(torch.ones(max_translation))).data.numpy()[0]
    else:
        x_shift, y_shift = 0, 0

    if random_line_width:
        line_width = pyro.sample("line_width",
                                 dist.uniform,
                                 Variable(torch.Tensor([0])),
                                 Variable(torch.Tensor([max_line_width]))).data.numpy()[0]

    if random_line_colour:
        line_rgb_colour = tuple(pyro.sample("line_{}".format(colour),
                                       dist.uniform,
                                       Variable(torch.Tensor([0])),
                                       Variable(torch.Tensor([1]))).data.numpy()[0]
                           for colour in ("red", "green", "blue"))
    else:
        line_rgb_colour = (0.2, 0.2, 0.8)

    if random_bar_width:
        bar_width = pyro.sample("bar_width",
                                dist.uniform,
                                Variable(torch.Tensor([0])),
                                Variable(torch.Tensor([1]))).data.numpy()[0]

    if random_colour:
        rgb_colour = tuple(pyro.sample(colour,
                                       dist.uniform,
                                       Variable(torch.Tensor([0])),
                                       Variable(torch.Tensor([1]))).data.numpy()[0]
                           for colour in ("red", "green", "blue"))
    else:
        rgb_colour = (0.2, 0.2, 0.8)

    num_bars = pyro.sample("num_bars",
                           dist.categorical,
                           ps=Variable(torch.Tensor(np.array([0., 0., 1., 1., 1.])/3)))
    num_bars = num_bars.data.numpy()[0]

    bar_heights = []
    for bar_num in range(num_bars):
        bar_height = pyro.sample("bar_height_{}".format(bar_num),
                                 dist.uniform,
                                 Variable(torch.Tensor([0])),
                                 Variable(torch.Tensor([max_height])))
        bar_heights.append(bar_height.data.numpy()[0])

    fig, ax = plt.subplots()
    ax.bar(range(num_bars),
           bar_heights,
           width=bar_width,
           color=rgb_colour,
           linewidth=line_width,
           edgecolor=line_rgb_colour,
           label="Bar")
    ax.set_ylim(0, max_height)

    # get the graph as a matrix
    fig = set_size_pixels(fig, (width, height))
    image = Variable(fig2tensor(fig))
    plt.close()

    # do the translation
    background = Variable(torch.ones(3,
                                     height+max_translation,
                                     width+max_translation)) * 255.0
    background[:, y_shift:y_shift+height, x_shift:x_shift+width] = image
    image = background

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
