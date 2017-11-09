
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


svi = pyro.infer.SVI(plot_reader.model,
                     plot_reader.guide,
                     pyro.optim.Adam({"lr": 0.0001}),
                     loss="ELBO")

for i, (images, true_heights) in enumerate(data_generator(num_batches=10)):
    print("loss is", svi.step(images))


# for _, (x, _) in enumerate(train_loader):
#     # wrap the mini-batch in a PyTorch Variable
#     x = Variable(x)
#     # do ELBO gradient and accumulate loss
#     epoch_loss += svi.step(x)
# print(epoch_loss)
