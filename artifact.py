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
    def __init__(self, name,
                 model_kwargs,
                 guide_kwargs,
                 model_and_guide_kwargs,
                 compiler_kwargs,
                 optimiser_kwargs):
        """
        this should only be run once - never upon reloading an old artifact
        as long as PersistentArtifact.load is used
        """
        self.name = name
        self.model_kwargs = model_kwargs
        self.model_kwargs.update(model_and_guide_kwargs)
        self.guide_kwargs = guide_kwargs
        self.guide_kwargs.update(model_and_guide_kwargs)
        self.compiler_kwargs = compiler_kwargs
        self.optimiser_kwargs = optimiser_kwargs

        self.validation_losses = []

        self._init_paths()

        self.training_steps = 0

        self.steps_at_last_make_plots = self.training_steps
        self.steps_at_last_make_videos = self.training_steps

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
        posterior_videos_path = "{}/posterior_videos".format(self.directory)
        os.makedirs(attention_graphics_path)
        os.makedirs(posterior_videos_path)

        self.paths = {"weights": weights_path,
                      "infer_log": inference_log_path,
                      "attention_graphics": attention_graphics_path,
                      "posterior_videos": posterior_videos_path}

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
        csis.set_model_args(**model_kwargs)
        csis.set_compiler_args(**self.compiler_kwargs)

        # Force validation batch to be created with a certain random seed
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        csis._init_compiler()
        torch.set_rng_state(rng_state)

        csis.iterations = self.training_steps
        csis.compile(optim, num_steps=N_STEPS, cuda=CUDA)

        torch.save(guide.state_dict(), self.paths["weights"])

        validation_log = csis.get_compile_log()["validation"]
        self.validation_losses.extend(validation_log)

        self.training_steps += N_STEPS
        self.save()

    def make_plots(self,
                   test_folder="default",
                   max_plots=np.inf:
                   cuda=False):
        if test_folder == "default":
            test_folder = "{}/test".format(DATASET_PATH)

        subprocess.check_call(["rm", "-f",
                               "{}/*".format(self.paths["attention_graphics"])])
        self.steps_at_last_make_plots = self.training_steps

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda
        guide_kwargs["attention_graphics_path"] = self.paths["attention_graphics"]
        guide_kwargs["collect_history"] = True
        guide = Guide(**guide_kwargs)
        guide.load_state_dict(torch.load(self.paths["weights"]))

        csis = CSIS(model=model,
                    guide=guide,
                    num_samples=1)
        csis.set_model_args(**model_kwargs)
        marginal = Marginal(csis)

        img_no = 0
        while img_no < max_plots and os.path.isfile("{}/graph_{}.png".format(test_folder, img_no)):
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

    def make_posterior_videos(self,
                              test_folder="default",
                              max_plots=np.inf,
                              cuda=False):
        if test_folder == "default":
            test_folder = "{}/test".format(DATASET_PATH)

        subprocess.check_call(["rm", "-f",
                               "{}/*".format(self.paths["posterior_videos"])])
        self.steps_at_last_make_videos = self.training_steps

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda
        guide = Guide(**guide_kwargs)
        guide.load_state_dict(torch.load(self.paths["weights"]))

        csis = CSIS(model=model,
                    guide=guide,
                    num_samples=10)
        csis.set_model_args(**model_kwargs)
        marginal = Marginal(csis)

        img_no = 0
        while img_no < max_plots and os.path.isfile("{}/graph_{}.png".format(test_folder, img_no)):
            print("running inference no.", img_no)
            image = Image.open("{}/graph_{}.png".format(test_folder, img_no))
            image = image2variable(image)
            weighted_traces = marginal.trace_dist._traces(observed_image=image)
            for trace_no, (trace, log_weight) in enumerate(weighted_traces):
                image = trace.nodes["_RETURN"]["value"]["image"]
                image = Image.fromarray(image.data.numpy())
                image.save("{}/{}.png".format(self.paths["posterior_videos"],
                                              trace_no))
            subprocess.check_call(["ffmpeg",
                                   "-r", "4",
                                   "-f", "image2",
                                   "-s", "200x200",
                                   "-i", "{}/%d.png".format(self.paths["posterior_videos"]),
                                   "-vcodec", "libx264",
                                   "-crf", "25",
                                   "-pix_fmt", "yuv420p",
                                   "{}/{}.mp4".format(self.paths["posterior_videos"],
                                                      img_no)])
            subprocess.check_call(["rm", "-f",
                                   "{}/*.png".format(self.paths["posterior_videos"])])
            img_no += 1

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
