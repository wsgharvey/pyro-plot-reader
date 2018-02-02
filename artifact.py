import torch
from torch.autograd import Variable

import pyro
from pyro.infer import CSIS
import pyro.distributions as dist

from model import model
from guide import Guide
from file_paths import ARTIFACT_FOLDER

import pickle
import os


class PersistentArtifact(object):
    def __init__(self, name, guide_kwargs, compiler_kwargs, optimiser_kwargs):
        """
        this should only be run once - never upon reloading an old artifact
        as long as PersistentArtifact.load is used
        """
        self.name = name
        self.guide_kwargs = guide_kwargs
        self.compiler_kwargs = compiler_kwargs
        self.optimiser_kwargs = optimiser_kwargs
        self.validation_losses = []

        self.directory = "{}/{}".format(ARTIFACT_FOLDER, name)
        if os.path.exists(self.directory):
            raise Exception("Folder already exists at {}".format(self.directory))
        else:
            os.makedirs(self.directory)

        self.weights_path = "{}/weights.pt".format(self.directory)

        self.save()

    def compile(self, N_STEPS, CUDA=False):
        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = CUDA
        guide = Guide(**guide_kwargs)

        try:
            guide.load_state_dict(torch.load(self.weights_path))
        except:
            pass

        if CUDA:
            guide.cuda()

        optim = torch.optim.Adam(guide.parameters(), **self.optimiser_kwargs)

        csis = CSIS(model=model,
                    guide=guide,
                    num_samples=1)
        csis.set_model_args()
        csis.set_compiler_args(**self.compiler_kwargs)

        csis.compile(optim, num_steps=N_STEPS, cuda=CUDA)

        torch.save(guide.state_dict(), self.weights_path)

        validation_log = csis.get_compile_log()["validation"]
        self.validation_losses.extend(validation_log)

        self.save()

    def infer(self, CUDA=False):

        self.save()

    def save(self):
        path = "{}/artifact.p".format(self.directory)
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(name):
        path = "{}/{}/artifact.p".format(ARTIFACT_FOLDER, name)
        return pickle.load(open(path, "rb"))

# a = PersistentArtifact("bobb", {}, {"kj", 34}, {})
# a.save()
#
# print("made 1, now reload it")
#
# b = PersistentArtifact.load("bobb")
# print(b.compiler_kwargs)
