import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
import pyro.infer as infer
import pyro.distributions as dist

from file_paths import ARTIFACT_PATH, DATASET_PATH
from model import model
from guide import Guide

from PIL import Image
import numpy as np


def expected_val(weighted_traces, attribute):
    """
    calculates the expected value of an attribute over weighted traces
    given in the form, [({attribute_dict_1}, weight_1), ...]
    """
    att_sum, weight_sum = 0, 0
    weighted_traces = list(weighted_traces)
    max_log_weight = max(float(log_weight.data.numpy()[0]) for _, log_weight in weighted_traces)
    for trace, log_weight in weighted_traces:
        ret = trace.nodes["_RETURN"]["value"][attribute]
        weight = (log_weight-max_log_weight).exp().data.numpy()[0]
        weight_sum += weight
        att_sum += ret * weight
    return att_sum/weight_sum


guide = Guide()
guide.load_state_dict(torch.load(ARTIFACT_PATH))

csis = infer.CSIS(model=model,
                  guide=guide,
                  num_samples=10)
csis.set_model_args()
marginal = infer.Marginal(csis)

predictions = []
for img_no in range(100):
    image = Image.open("{}/test/graph_{}.png".format(DATASET_PATH, img_no))
    image = np.array(image).astype(np.float32)
    image = np.array([image[..., 0], image[..., 1], image[..., 2]])
    image = Variable(torch.Tensor(image))

    marg = marginal.trace_dist._traces(observed_image=image)

    predictions.append(expected_val(marg, "bar_heights"))

with open("{}/test_predictions.csv".format(DATASET_PATH), 'w') as f:
    f.write("\n".join((map(lambda x: ",".join(map(str, x)), predictions))))
