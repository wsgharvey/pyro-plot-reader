
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from plot_reader import PlotReader

import numpy as np

from PIL import Image


plot_reader = PlotReader()


def data_generator(num_batches=-1, batch_size=8, use_cuda=False):
    """ generator for stream of data batches
        yields tuples of (
                            batch_size x 3 x height x width Variable of image
                            batch_size variable of true bar heights
        )
    """
    plot_reader = PlotReader()
    while True:
        images = Variable(torch.zeros(batch_size,
                                      200,
                                      200,
                                      3).type(torch.FloatTensor))
        true_heights = Variable(torch.zeros(batch_size).type(torch.FloatTensor))

        for i in range(batch_size):
            true_heights[i], images[i] = plot_reader.model(None,
                                                           return_image=True)

        yield images, true_heights


svi = pyro.infer.SVI(plot_reader.test_model,
                     plot_reader.guide,
                     pyro.optim.Adam({"lr": 0.01}),
                     loss="ELBO",
                     num_particles=5)

# images, true_heights = next(data_generator())
# image, true_height = images[0], true_heights[0]
# print("height should be", true_height)
#
# from PIL import Image
# img = Image.fromarray(image.data.numpy().astype(np.uint8), 'RGB')
# img.show()


while True:
    print("loss is", svi.step())
    print("\n")
