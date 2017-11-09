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

from helpers import fig2tensor,\
                    set_size_pixels
from nn import BarHeightNet


class PlotReader(nn.Module):
    def __init__(self):
        super(PlotReader, self).__init__()
        self.bar_height_net = BarHeightNet()

    def model(self, observed_image, return_image=True):
        """ generates a bar chart with a single bar
        """
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
        # when running inference
        if return_image:
            return bar_height, image
        else:
            return bar_height

    def guide(self, observed_image):
        mean_height = self.bar_height_net(observed_image)
        return pyro.sample("bar_height",
                           dist.normal,
                           mean_height,
                           Variable(torch.ones(mean_height.size())))

# def bar_height_guide():
#     """ a cheeky guide that has no effect
#     """
#     mean_height = pyro.param("mean_height",
#                              Variable(torch.Tensor([4]),
#                                       requires_grad=True))
#     height_std = pyro.param("height_std",
#                             Variable(torch.Tensor([7]),
#                                      requires_grad=True))
#     return pyro.sample("bar_height",
#                        dist.normal,
#                        mean_height,
#                        height_std)


# # make data:
# real_height, real_img = graph(return_image=True)
# print("true height is", real_height)
#
# # condition on the data
# conditioned_graph = pyro.condition(
#                         graph,
#                         data={"observed_image": real_img.view(-1)})
#
# # optimise guide parameters
# svi = pyro.infer.SVI(model=conditioned_graph,
#                      guide=bar_height_guide,
#                      optim=pyro.optim.SGD({"lr": 0.1}),
#                      loss="ELBO")
# for i in range(100):
#     print("\n\nPARAMS after {}:".format(i), list(pyro.get_param_store().named_parameters()))
#     svi.step()
#
# # run inference
# posterior = pyro.infer.Importance(conditioned_graph,
#                                   guide=bar_height_guide,
#                                   num_samples=10)
# marginal = pyro.infer.Marginal(posterior)
#
# # sample from empirical posterior
# print("estimated height is:\n", marginal())
