import torch
from torch.autograd import Variable

import pyro
from pyro.infer import CSIS, Marginal
import pyro.distributions as dist
from pyro.poutine.trace import Trace
import pyro.poutine as poutine

from model import Model
from guide import Guide
from file_paths import ARTIFACT_FOLDER, DATASET_FOLDER

from helpers import image2variable

from PIL import Image
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

import pickle
import os
import subprocess


class PersistentArtifact(object):
    def __init__(self, name,
                 model_kwargs,
                 guide_kwargs,
                 compiler_kwargs,
                 optimiser_kwargs):
        """
        this should only be run once - never upon reloading an old artifact
        as long as PersistentArtifact.load is used
        """
        self.name = name
        self.model_kwargs = model_kwargs
        self.guide_kwargs = guide_kwargs
        self.compiler_kwargs = compiler_kwargs
        self.optimiser_kwargs = optimiser_kwargs

        self.validation_losses = []

        self._init_paths()

        self.training_steps = 0

        self.steps_at_last_make_plots = self.training_steps
        self.steps_at_last_make_videos = self.training_steps

        self.save()

    def _init_paths(self, create_dirs=True):
        self.directory = "{}/{}".format(ARTIFACT_FOLDER, self.name)
        if os.path.exists(self.directory):
            raise Exception("Folder already exists at {}".format(self.directory))
        else:
            os.makedirs(self.directory)

        weights_path = "{}/weights.pt".format(self.directory)
        inference_log_path = "{}/infer_log.p".format(self.directory)
        attention_graphics_path = "{}/attention_graphics".format(self.directory)
        posterior_videos_path = "{}/posterior_videos".format(self.directory)
        if create_dirs:
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

        csis = CSIS(model=Model(**self.model_kwargs),
                    guide=guide,
                    num_samples=1)
        csis.set_model_args()
        csis.set_compiler_args(**self.compiler_kwargs)

        # Force validation batch to be created with a certain random seed
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        csis._init_compiler()
        torch.set_rng_state(rng_state)

        csis.iterations = self.training_steps
        csis.compile(optim=optim, num_steps=N_STEPS, cuda=CUDA)

        torch.save(guide.state_dict(), self.paths["weights"])

        validation_log = csis.get_compile_log()["validation"]
        self.validation_losses.extend(validation_log)

        self.training_steps += N_STEPS
        self.save()

    def infer(self,
              dataset_name,
              attention_plots=True,
              start_no=0,
              max_plots=np.inf,
              num_traces=10,
              cuda=False):
        test_folder = "{}/{}/test".format(DATASET_FOLDER, dataset_name)
        target = []
        with open("{}/targets.csv".format(test_folder), 'r') as f:
            ground_truths = f.read().splitlines()
            ground_truths = [eval('[' + line + ']') for line in ground_truths]

        subprocess.check_call(["rm", "-f",
                               "{}/*".format(self.paths["attention_graphics"])])
        self.steps_at_last_make_plots = self.training_steps

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda
        if attention_plots:
            guide_kwargs["attention_graphics_path"] = self.paths["attention_graphics"]
        else:
            guide_kwargs["attention_graphics_path"] = None
        guide_kwargs["collect_history"] = True
        guide = Guide(**guide_kwargs)
        guide.load_state_dict(torch.load(self.paths["weights"]))

        text = ""
        img_no = start_no
        dataset_log_pdf = 0
        while img_no < start_no + max_plots and os.path.isfile("{}/graph_{}.png".format(test_folder, img_no)):
            if start_no == 0:
                open("{}/confidence_intervals".format(self.directory), "w")

            print("running inference no.", img_no)
            image = Image.open("{}/graph_{}.png".format(test_folder, img_no))
            image = image2variable(image)

            true_data = ground_truths[img_no]
            if type(true_data[0]) is float:
                # then there is no multi_bar_charts option
                num_bar_charts = None
                num_bars = len(true_data)
            else:
                # there is a multi_bar_charts option
                num_bar_charts = len(true_data)
                num_bars = len(true_data[0])

            log_pdfs = []
            log_pdfs.append(self.log_pdf(num_bar_charts, num_bars, true_data, guide, observed_image=image, print_params=True).data.numpy())
            for t in range(1, num_traces):
                print(t)
                guide = Guide(**guide_kwargs)
                guide.load_state_dict(torch.load(self.paths["weights"]))
                log_pdfs.append(self.log_pdf(num_bar_charts, num_bars, true_data, guide, observed_image=image, print_params=False).data.numpy())
            log_pdf = logsumexp(log_pdfs) - np.log(num_traces)

            if log_pdf > -10000000 and log_pdf < 10000000:
                dataset_log_pdf += log_pdf
            else:
                raise IndexError

            datum_history = guide.get_history()[-num_traces:]
            # print("NUM BARS:", guide.get_history()[-T:][-1])
            example = datum_history[0]
            bar_height_sample_names = [name for name in example if name[:10] == 'bar_height']
            text += "inference on data point {}:\n".format(img_no)

            for sample_name in bar_height_sample_names:
                bar_height_predictions = []
                for trace in datum_history:
                    params = trace[sample_name]
                    mode, cert = params
                    mode, cert = mode.item(), cert.item()
                    norm_m = mode
                    norm_c = cert + 2
                    dist = beta(norm_m * (norm_c - 2),
                                (1 - norm_m) * (norm_c - 2))
                    bar_height_predictions.append(dist)
                confidence_intervals = []
                # find where sum of cdfs in predictions add up to 0.05*T and 0.095*T
                for target in [0.025*num_traces, 0.975*num_traces]:
                    lower, upper = 0., 1.
                    for _ in range(16):
                        guess = (lower+upper)/2
                        cdf = sum(dist.cdf(guess) for dist in bar_height_predictions)
                        if cdf > target:
                            upper = guess
                        else:
                            lower = guess
                    confidence_intervals.append(guess)
                text += sample_name + ": " + str(confidence_intervals) + "\n"
            img_no += 1

        inference_log = guide.get_history()

        with open("{}/confidence_intervals".format(self.directory), "a") as f:
            f.write(text)

        pickle.dump(inference_log, open(self.paths["infer_log"], 'wb'))
        return dataset_log_pdf

    def make_posterior_videos(self,
                              dataset_name,
                              max_plots=np.inf,
                              start_no=0,
                              cuda=False):
        test_folder = "{}/{}/test".format(DATASET_FOLDER, dataset_name)

        subprocess.check_call(["rm", "-f",
                               "{}/*".format(self.paths["posterior_videos"])])

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda


        img_no = start_no
        while img_no < max_plots and os.path.isfile("{}/graph_{}.png".format(test_folder, img_no)):

            for trace_no in range(10):
                guide = Guide(**guide_kwargs)
                guide.load_state_dict(torch.load(self.paths["weights"]))

                csis = CSIS(model=Model(**self.model_kwargs),
                            guide=guide,
                            num_samples=1)
                csis.set_model_args()
                marginal = Marginal(csis)

                print("running inference no.", img_no)
                image = Image.open("{}/graph_{}.png".format(test_folder, img_no))
                image = image2variable(image)
                weighted_traces = marginal.trace_dist._traces(observed_image=image)
                trace, log_weight = next(weighted_traces)
                image = trace.nodes["_RETURN"]["value"]["image"]
                image = image.view(3, 210, 210)
                image = image.data.numpy()
                imgArray = np.zeros((210, 210, 3), 'uint8')
                imgArray[..., 0] = image[0]
                imgArray[..., 1] = image[1]
                imgArray[..., 2] = image[2]
                image = Image.fromarray(imgArray)
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

    def posterior_samples(self,
                          dataset_name,
                          max_plots=np.inf,
                          start_no=0,
                          n_traces=10,
                          cuda=False):
        test_folder = "{}/{}/test".format(DATASET_FOLDER, dataset_name)

        subprocess.check_call(["rm", "-f",
                               "{}/*".format(self.paths["posterior_videos"])])

        guide_kwargs = self.guide_kwargs.copy()
        guide_kwargs["cuda"] = cuda

        with torch.no_grad():
            img_no = start_no
            while img_no < max_plots+start_no and os.path.isfile("{}/graph_{}.png".format(test_folder, img_no)):

                log_weights = []
                images = []
                for trace_no in range(n_traces):
                    print(trace_no) 
                    guide = Guide(**guide_kwargs)
                    guide.load_state_dict(torch.load(self.paths["weights"]))

                    csis = CSIS(model=Model(**self.model_kwargs),
                                guide=guide,
                                num_samples=1)
                    csis.set_model_args()
                    marginal = Marginal(csis)

                    print("running inference no.", img_no)
                    image = Image.open("{}/graph_{}.png".format(test_folder, img_no))
                    image = Variable(image2variable(image).data, volatile=True)
                    weighted_traces = marginal.trace_dist._traces(observed_image=image)
                    trace, log_weight = next(weighted_traces)
                    image = trace.nodes["_RETURN"]["value"]["image"]
                    image = image.view(3, 210, 210)
                    image = image.data.numpy()
                    imgArray = np.zeros((210, 210, 3), 'uint8')
                    imgArray[..., 0] = image[0]
                    imgArray[..., 1] = image[1]
                    imgArray[..., 2] = image[2]
                    # image = Image.fromarray(imgArray)
                    images.append(imgArray)
                    log_weights.append(log_weight)
                    #guide.detach_hidden_states()

                log_weights = torch.Tensor(log_weights)
                weights = torch.nn.Softmax(dim=0)(log_weights)
                print(weights)
                composite_image = np.zeros((210, 210, 3), 'float32')
                for image, weight in zip(images, weights):
                    composite_image += image*weight
                composite_image = composite_image.astype('uint8')
                composite_image = Image.fromarray(composite_image)
                composite_image.save("{}/posterior_{}.png".format(self.paths["posterior_videos"],
                                                              img_no))
                img_no += 1

    def save(self):
        path = "{}/artifact.p".format(self.directory)
        pickle.dump(self, open(path, 'wb'))

    def copy(self, new_name):
        new = type(self)(new_name, None, None, None, None)
        subprocess.check_call(["rm", "-rf",
                               new.directory])
        new.__dict__.update(self.__dict__)
        new.name = new_name
        new._init_paths(create_dirs=False)
        for path_name in self.paths:
            subprocess.check_call(["cp", "-r",
                                   self.paths[path_name],
                                   new.paths[path_name]])
        new.save()

    def log_pdf(self, num_bar_charts, num_bars, bar_heights, guide, *args, **kwargs):
        """ assesses log prob of the guide on a set of samples
        """
        # make model trace
        ground_truth_trace = Trace()
        if num_bar_charts is not None:
            ground_truth_trace.add_node("num_bar_charts",
                                        type="sample",
                                        name="num_bar_charts",
                                        is_observed=False,
                                        value=Variable(torch.Tensor([num_bar_charts])).type(torch.IntTensor),
                                        args=(), kwargs={})
        ground_truth_trace.add_node("num_bars",
                                    type="sample",
                                    name="num_bars",
                                    is_observed=False,
                                    value=Variable(torch.Tensor([num_bars])).type(torch.IntTensor),
                                    args=(), kwargs={})
        for i, bar_height in enumerate(bar_heights):
            if type(bar_height) is float:
                ground_truth_trace.add_node("bar_height_{}".format(i),
                                            type="sample",
                                            name="num_bars",
                                            is_observed=False,
                                            value=Variable(torch.Tensor([bar_height])),
                                            args=(), kwargs={})
            else:
                for j, bar_height in enumerate(bar_height):
                    ground_truth_trace.add_node("bar_height_{}_{}".format(i, j),
                                                type="sample",
                                                name="num_bars",
                                                is_observed=False,
                                                value=Variable(torch.Tensor([bar_height])),
                                                args=(), kwargs={})

        # run guide against trace
        guide_trace = poutine.trace(
            poutine.replay(guide, ground_truth_trace)).get_trace(*args, **kwargs)   # does observed_image go into kwargs?

        # calculate log pdf of every sample in guide)
        guide_trace.log_pdf()

        # sum log pdf of relevant samples
        num_bars_log_pdf = 0
        bar_heights_log_pdf = 0
        observes_log_pdf = 0
        for name, site in guide_trace.nodes.items():
            if name == "num_bars":
                num_bars_log_pdf += site["log_pdf"]
            if name[:10] == "bar_height":
                bar_heights_log_pdf += site["log_pdf"]
            if site['type'] == 'observe':
                observes_log_pdf += site["log_pdf"]
        return num_bars_log_pdf + bar_heights_log_pdf + observes_log_pdf

    @staticmethod
    def load(name, artifact_folder=None):
        if artifact_folder is None:
            artifact_folder = ARTIFACT_FOLDER
        path = "{}/{}/artifact.p".format(artifact_folder, name)
        return pickle.load(open(path, "rb"))
