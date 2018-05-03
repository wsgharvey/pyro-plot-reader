import pyro
import pyro.infer
import pyro.distributions as dist

import torch
from torch.autograd import Variable

import numpy as np
from scipy.misc import imresize

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
                 scale="fixed",
                 multi_bar_charts=False,
                 random_img_dim=False,
                 random_layout=False):
        self.random_colour = random_colour
        self.random_bar_width = random_bar_width
        self.random_line_colour = random_line_colour
        self.random_line_width = random_line_width
        self.wiggle_picture = wiggle_picture
        self.scale = scale
        self.multi_bar_charts = multi_bar_charts
        self.random_img_dim = random_img_dim
        self.random_layout = random_layout

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
            max_height = max_heights[int(index.data.numpy())]
        elif self.scale == "continuous":
            max_max_height = 100
            max_height = pyro.sample("max_height",
                                     dist.uniform,
                                     Variable(torch.Tensor([0])),
                                     Variable(torch.Tensor([max_max_height]))).data.numpy()[0]
        elif self.scale == "very_general":
            max_max_height = 320
            max_height_log = pyro.sample("max_log_height",
                                         dist.uniform,
                                         Variable(torch.Tensor([-1])),
                                         Variable(torch.Tensor([2.5]))).data.numpy()[0]
            max_height = 10**max_height_log
        else:
            raise Exception("scale argument not valid")

        if self.multi_bar_charts:
            num_bar_charts = pyro.sample("num_bar_charts",
                                         dist.categorical,
                                         ps=Variable(torch.Tensor([0, 1., 1., 1., 1.])))
            num_bar_charts = int(num_bar_charts)
        else:
            num_bar_charts = 1

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
        bar_width = bar_width / num_bar_charts

        if self.random_colour:
            if self.multi_bar_charts:
                rgb_colours = [tuple(pyro.sample("{}_{}".format(colour, i),
                                                 dist.uniform,
                                                 Variable(torch.Tensor([0])),
                                                 Variable(torch.Tensor([1]))).data.numpy()[0]
                                     for colour in ("red", "green", "blue"))
                               for i in range(num_bar_charts)]
            else:
                rgb_colour = tuple(pyro.sample(colour,
                                               dist.uniform,
                                               Variable(torch.Tensor([0])),
                                               Variable(torch.Tensor([1]))).data.numpy()[0]
                                   for colour in ("red", "green", "blue"))
        else:
            if self.multi_bar_charts:
                rgb_colours = [(0.2, 0.2, 0.8) for _ in range(num_bar_charts)]
            else:
                rgb_colour = (0.2, 0.2, 0.8)

        num_bars = int(pyro.sample("num_bars",
                                   dist.categorical,
                                   ps=Variable(torch.Tensor(np.array([0., 0., 1., 1., 1.])/3))))

        if self.random_img_dim:
            img_width = 150 + num_bars*pyro.sample("img_width",
                                                   dist.uniform,
                                                   Variable(torch.Tensor([25])),
                                                   Variable(torch.Tensor([100]))).data.numpy()[0]
            img_height = pyro.sample("img_height",
                                     dist.uniform,
                                     Variable(torch.Tensor([200])),
                                     Variable(torch.Tensor([500]))).data.numpy()[0]
            img_width, img_height = int(img_width), int(img_height)
        else:
            img_width, img_height = width, height

        if num_bar_charts > 1:
            density = pyro.sample("density",
                                  dist.uniform,
                                  Variable(torch.Tensor([0])),
                                  Variable(torch.Tensor([1]))).data.numpy()[0]

            legend = int(pyro.sample("legend",
                                     dist.categorical,
                                     ps=Variable(torch.ones(2))))
            legend = bool(legend)
        else:
            legend = False

        if self.random_layout:
            no_spines = int(pyro.sample("no_spines",
                                        dist.categorical,
                                        ps=Variable(torch.ones(2))))
            no_spines = bool(legend)
        else:
            no_spines = False

        if self.multi_bar_charts:
            bar_heights = []
            for bar_chart in range(num_bar_charts):
                bar_heights.append([])
                for bar_num in range(num_bars):
                    bar_height = pyro.sample("bar_height_{}_{}".format(bar_chart, bar_num),
                                             dist.uniform,
                                             Variable(torch.Tensor([0])),
                                             Variable(torch.Tensor([max_height])))
                    bar_heights[-1].append(bar_height.data.numpy()[0])
        else:
            bar_heights = []
            for bar_num in range(num_bars):
                bar_height = pyro.sample("bar_height_{}".format(bar_num),
                                         dist.uniform,
                                         Variable(torch.Tensor([0])),
                                         Variable(torch.Tensor([max_height])))
                bar_heights.append(bar_height.data.numpy()[0])

        fig, ax = plt.subplots()
        if self.multi_bar_charts:
            for bar_chart in range(num_bar_charts):
                if num_bar_charts > 1:
                    loose = bar_chart/(num_bar_charts)-0.5*(num_bar_charts-1)/num_bar_charts
                    dense = bar_chart*bar_width - num_bar_charts*bar_width/2
                    x_bar_offset = density*dense + (1-density)*loose
                else:
                    x_bar_offset = 0

                ax.bar(np.arange(num_bars) + x_bar_offset,
                       bar_heights[bar_chart],
                       width=bar_width,
                       color=rgb_colours[bar_chart],
                       linewidth=line_width,
                       edgecolor=line_rgb_colour,
                       label="bar_chart_{}".format(bar_chart))
        else:
            ax.bar(np.arange(num_bars),
                   bar_heights,
                   width=bar_width,
                   color=rgb_colour,
                   linewidth=line_width,
                   edgecolor=line_rgb_colour)
        ax.set_ylim(0, max_height)
        if legend:
            ax.legend()
        if no_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # get the graph as a matrix
        fig = set_size_pixels(fig, (img_width, img_height))
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

        if self.multi_bar_charts:
            return {"image": image,
                    "bar_heights": list(map(str, bar_heights))}
        else:
            return {"image": image,
                    "bar_heights": np.array(bar_heights)}
