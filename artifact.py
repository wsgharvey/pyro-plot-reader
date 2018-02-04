import torch
from torch.autograd import Variable

import pyro
from pyro.infer import CSIS, Marginal
import pyro.distributions as dist

from model import model
from guide import Guide
from file_paths import ARTIFACT_FOLDER, DATASET_PATH

# these will be unneccessary one image processing is moved
from PIL import Image
import numpy as np

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
        self.inference_log_path = "{}/infer_log.p".format(self.directory)

        self.training_steps = 0

        self.save()

    def compile(self, N_STEPS, CUDA=False):
        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = CUDA
        guide = Guide(**guide_kwargs)

        try:
            guide.load_state_dict(torch.load(self.weights_path))
        except:
            pass

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

        self.training_steps += N_STEPS
        self.save()

    def make_plots(self,
                   test_folder="default",
                   cuda=False):
        if test_folder == "default":
            test_folder = "{}/test".format(DATASET_PATH)

        attention_graphics_path = "{}/attention_graphics".format(self.directory)
        if not os.path.exists(attention_graphics_path):
            os.makedirs(attention_graphics_path)

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda
        guide_kwargs["attention_graphics_path"] = attention_graphics_path
        guide_kwargs["collect_history"] = True
        guide = Guide(**guide_kwargs)

        csis = CSIS(model=model,
                    guide=guide,
                    num_samples=1)
        csis.set_model_args()
        marginal = Marginal(csis)

        img_no = 0
        while True:
            try:
                image = Image.open("{}/graph_{}.png".format(test_folder, img_no))
            except OSError:
                break
            # TODO: move this image processing to helpers
            image = np.array(image).astype(np.float32)
            image = np.array([image[..., 0], image[..., 1], image[..., 2]])
            image = Variable(torch.Tensor(image))

            weighted_traces = marginal.trace_dist._traces(observed_image=image)

            for trace, log_weight in weighted_traces:
                pass
            img_no += 1

        inference_log = guide.get_history()
        pickle.dump(inference_log, open(self.inference_log_path, 'wb'))

    def save(self):
        path = "{}/artifact.p".format(self.directory)
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(name):
        path = "{}/{}/artifact.p".format(ARTIFACT_FOLDER, name)
        return pickle.load(open(path, "rb"))
