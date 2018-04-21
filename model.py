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


class Model(object):
    def __init__(self,
                 random_colour=True,
                 random_bar_width=True,
                 random_line_colour=True,
                 random_line_width=True,
                 wiggle_picture=False,
                 scale="fixed"):
        self.random_colour = random_colour
        self.random_bar_width = random_bar_width
        self.random_line_colour = random_line_colour
        self.random_line_width = random_line_width
        self.wiggle_picture = wiggle_picture
        self.scale = scale

    def __call__(self, observed_image=Variable(torch.zeros(200, 200))):
        max_height = 10
        max_line_width = 2.5
        max_translation = 10
        height, width = 200, 200

        if self.wiggle_picture:
            x_shift = int(pyro.sample("x_shift",
                                      dist.categorical,
                                      ps=Variable(torch.ones(max_translation))))

            y_shift = int(pyro.sample("y_shift",
                                      dist.categorical,
                                      ps=Variable(torch.ones(max_translation))))
        else:
            x_shift, y_shift = 0, 0

        if self.scale == "fixed":
            max_height = 10
        elif self.scale == "discrete":
            max_heights = [10, 50, 100]
            index = pyro.sample("max_height",
                                dist.categorical,
                                ps=Variable(torch.ones(3)))
            max_height = max_heights[int(index.data.numpy()[0])]
        elif self.scale == "continuous":
            max_max_height = 100
            max_height = pyro.sample("max_height",
                                     dist.uniform,
                                     Variable(torch.Tensor([0])),
                                     Variable(torch.Tensor([max_max_height]))).data.numpy()[0]
        else:
            raise Exception("scale argument not valid")

        if self.random_line_width:
            line_width = pyro.sample("line_width",
                                     dist.uniform,
                                     Variable(torch.Tensor([0])),
                                     Variable(torch.Tensor([max_line_width]))).data.numpy()[0]
        else:
            line_width = 0

        if self.random_line_colour:
            line_rgb_colour = tuple(pyro.sample("line_{}".format(colour),
                                                dist.uniform,
                                                Variable(torch.Tensor([0])),
                                                Variable(torch.Tensor([1]))).data.numpy()[0]
                               for colour in ("red", "green", "blue"))
        else:
            line_rgb_colour = (0, 0, 0)

        if self.random_bar_width:
            bar_width = pyro.sample("bar_width",
                                    dist.uniform,
                                    Variable(torch.Tensor([0])),
                                    Variable(torch.Tensor([1]))).data.numpy()[0]
        else:
            bar_width = 0.8

        if self.random_colour:
            rgb_colour = tuple(pyro.sample(colour,
                                           dist.uniform,
                                           Variable(torch.Tensor([0])),
                                           Variable(torch.Tensor([1]))).data.numpy()[0]
                               for colour in ("red", "green", "blue"))
        else:
            rgb_colour = (0.2, 0.2, 0.8)

        num_bars = int(pyro.sample("num_bars",
                                   dist.categorical,
                                   ps=Variable(torch.Tensor(np.array([0., 0., 1., 1., 1.])/3))))

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
