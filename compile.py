import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
from pyro.infer import CSIS
import pyro.distributions as dist

from file_paths import ARTIFACT_PATH
from model import model
from guide import Guide

torch.manual_seed(4)

guide = Guide()
guide.load_state_dict(torch.load(ARTIFACT_PATH))

optim = torch.optim.Adam(guide.parameters(), lr=1e-6)

csis = CSIS(model=model,
            guide=guide,
            num_samples=1)
csis.set_model_args()
csis.set_compiler_args(num_particles=8)

csis.compile(optim, num_steps=500)

torch.save(guide.state_dict(), ARTIFACT_PATH)
