
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
import pyro.infer as infer
import pyro.distributions as dist

from plot_reader import PlotReader

import numpy as np

from PIL import Image

torch.manual_seed(0)

plot_reader = PlotReader()

csis = infer.CSIS(model=plot_reader.model,
                  optim=torch.optim.Adam)
csis.compile(num_steps=1000,
             num_particles=8)

# posterior = csis.get_posterior(num_samples=4)
