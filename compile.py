import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
from pyro.infer import CSIS
import pyro.distributions as dist

from file_paths import ARTIFACT_PATH, COMPILE_LOG_PATH
from model import model
from guide import Guide

NEW_ARTIFACT = True
N_STEPS = 2000
CUDA = True 

torch.manual_seed(0)

guide = Guide()
if not NEW_ARTIFACT:
    guide.load_state_dict(torch.load(ARTIFACT_PATH))
if CUDA:
    guide.cuda()

optim = torch.optim.Adam(guide.parameters(), lr=1e-6)

csis = CSIS(model=model,
            guide=guide,
            num_samples=1)
csis.set_model_args()
csis.set_compiler_args(num_particles=8)

csis.compile(optim, num_steps=N_STEPS, cuda=CUDA)

torch.save(guide.state_dict(), ARTIFACT_PATH)

log = csis.get_compile_log()
mode = 'w' if NEW_ARTIFACT else 'a'
with open(COMPILE_LOG_PATH, mode) as f:
    losses = log["validation"]
    string = "\n".join(map(",".join, map(lambda x: map(str, x), losses))) + "\n"
    f.write(string)
