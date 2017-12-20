
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
import pyro.infer as infer
import pyro.distributions as dist

from model import model
from guide import Guide

import numpy as np

from PIL import Image

torch.manual_seed(4)

guide = Guide()
optim = torch.optim.Adam(guide.parameters(), lr=5e-3)
# guide.load_state_dict(torch.load("./artifact.pt"))

csis = infer.CSIS(model=model,
                  guide=guide,
                  num_samples=5)
csis.set_model_args()
csis.set_compiler_args(num_particles=8)

csis.compile(optim, num_steps=1000)

torch.save(guide.state_dict(), "./artifact.pt")
