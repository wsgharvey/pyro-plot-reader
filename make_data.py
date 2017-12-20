#
# import torch
# from torch.autograd import Variable
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
#
# import pyro
# import pyro.infer as infer
# import pyro.distributions as dist

from pyro.infer.csis.prior import sample_from_prior

from model import model

from PIL import Image
import numpy as np


def create_dataset(file_path,
                   n_data):
    targets = []

    for i in range(n_data):
        trace = sample_from_prior(model)
        targets.append(trace.nodes["bar_height"]["value"].data.numpy()[0])
        img = trace.nodes["_RETURN"]["value"]
        img = img.view(3, 200, 200)
        img = img.data.numpy()

        imgArray = np.zeros((200, 200, 3), 'uint8')
        imgArray[..., 0] = img[0]
        imgArray[..., 1] = img[1]
        imgArray[..., 2] = img[2]

        img = Image.fromarray(imgArray)
        img.save(file_path + "/graph_{}.png".format(i))

    with open(file_path + "/targets.csv", 'w') as f:
        f.write("\n".join(map(str, targets)))


create_dataset("../data/bar-1d/train", 1000)
create_dataset("../data/bar-1d/validation", 100)
create_dataset("../data/bar-1d/test", 100)
