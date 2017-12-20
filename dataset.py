import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from plot_reader import PlotReader

import numpy as np


# def create_data(file_path, n_graphs):
#     plot_generator = PlotReader().model
#     heights = []
#     num_length = len(str(n_graphs-1))
#     for i in range(n_graphs):
#         true_height = plot_generator(None,
#                                      save_address="{}/graph_{}.pdf".format(file_path,
#                                                                            str(i).zfill(num_length)))
#         heights.append(str(round(true_height.data[0], 3)))
#
#     with open("{}/targets.csv".format(file_path),
#               'w') as f:
#         f.write("\n".join(heights)+"\n")
#
#
# create_data("/home/will/Documents/4yp/plot-reader/data/bar-1d/validation",
#             n_graphs=100)
# create_data("/home/will/Documents/4yp/plot-reader/data/bar-1d/test",
#             n_graphs=100)
# create_data("/home/will/Documents/4yp/plot-reader/data/bar-1d/train",
#             n_graphs=1000)
