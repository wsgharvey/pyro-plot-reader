import torch
from torch.autograd import Variable

import pyro
from pyro.infer import CSIS, Marginal
import pyro.distributions as dist

from model import model
from guide import Guide
from file_paths import ARTIFACT_FOLDER, DATASET_PATH

from helpers import image2variable

from PIL import Image
import numpy as np

import pickle
import os
import subprocess


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

        self._init_paths()

        self.training_steps = 0

        self.save()

    def _init_paths(self):
        self.directory = "{}/{}".format(ARTIFACT_FOLDER, self.name)
        if os.path.exists(self.directory):
            raise Exception("Folder already exists at {}".format(self.directory))
        else:
            os.makedirs(self.directory)

        weights_path = "{}/weights.pt".format(self.directory)
        inference_log_path = "{}/infer_log.p".format(self.directory)
        attention_graphics_path = "{}/attention_graphics".format(self.directory)
        os.makedirs(attention_graphics_path)

        self.paths = {"weights": weights_path,
                      "infer_log": inference_log_path,
                      "attention_graphics": attention_graphics_path}

    def compile(self, N_STEPS, CUDA=False):
        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = CUDA
        guide = Guide(**guide_kwargs)

        try:
            guide.load_state_dict(torch.load(self.paths["weights"]))
        except:
            pass

        optim = torch.optim.Adam(guide.parameters(), **self.optimiser_kwargs)

        csis = CSIS(model=model,
                    guide=guide,
                    num_samples=1)
        csis.set_model_args()
        csis.set_compiler_args(**self.compiler_kwargs)

        csis.compile(optim, num_steps=N_STEPS, cuda=CUDA)

        torch.save(guide.state_dict(), self.paths["weights"])

        validation_log = csis.get_compile_log()["validation"]
        self.validation_losses.extend(validation_log)

        self.training_steps += N_STEPS
        self.save()

    def make_plots(self,
                   test_folder="default",
                   cuda=False):
        if test_folder == "default":
            test_folder = "{}/test".format(DATASET_PATH)

        subprocess.check_call(["rm", "-f",
                               self.paths["attention_graphics"]])

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda
        guide_kwargs["attention_graphics_path"] = self.paths["attention_graphics"]
        guide_kwargs["collect_history"] = True
        guide = Guide(**guide_kwargs)

        csis = CSIS(model=model,
                    guide=guide,
                    num_samples=1)
        csis.set_model_args()
        marginal = Marginal(csis)

        img_no = 0
        while os.path.isfile("{}/graph_{}.png".format(test_folder, img_no)):
            print("running inference no.", img_no)
            image = Image.open("{}/graph_{}.png".format(test_folder, img_no))
            image = image2variable(image)
            weighted_traces = marginal.trace_dist._traces(observed_image=image)
            for trace, log_weight in weighted_traces:
                pass
            img_no += 1

        print("Improving attention graphics...")
        while os.path.isfile("{}/{}.png".format(self.paths["attention_graphics"], img_no)):
            img = Image.open("{}/graph_{}.png".format(test_folder, img_no))
            att = Image.open("{}/{}.png".format(self.paths["attention_graphics"], img_no)).convert('L')
            h, w, _ = np.asarray(img).shape
            black = Image.new("RGB", (h, w))
            black.paste(img, mask=att)
            img = np.asarray(black).copy()
            img *= int(255 / np.amax(img))
            img = Image.fromarray(img)
            img.save("{}/{}.png".format(self.paths["attention_graphics"], img_no))

        inference_log = guide.get_history()
        pickle.dump(inference_log, open(self.paths["infer_log"], 'wb'))

    def save(self):
        path = "{}/artifact.p".format(self.directory)
        pickle.dump(self, open(path, 'wb'))

    def copy(self, new_name):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.name = new_name
        new._init_paths()
        for path_name in self.paths:
            subprocess.check_call(["cp", "-3",
                                   self.paths[path_name],
                                   new.paths[path_name]])
        new.save()

    @staticmethod
    def load(name):
        path = "{}/{}/artifact.p".format(ARTIFACT_FOLDER, name)
        return pickle.load(open(path, "rb"))
