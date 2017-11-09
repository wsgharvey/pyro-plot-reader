
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


plot_reader = PlotReader()

# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


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
                                      3,
                                      200,
                                      200).type(torch.FloatTensor))
        true_heights = Variable(torch.zeros(batch_size).type(torch.FloatTensor))

        for i in range(batch_size):
            true_heights[i], images[i] = plot_reader.model(None,
                                                           return_image=True)

        yield images, true_heights

a, b = next(data_generator())
from PIL import Image
a = a[0].data.numpy()
a = np.transpose(a, (1, 2, 0))
print(a.shape)
a = Image.fromarray(a, 'RGB')
a.show()
# for i, j in enumerate(data_generator()):
#     print("i", i)
#     print("j", j)

# svi = pyro.infer.SVI(plot_reader.model,
#                      plot_reader.guide,
#                      pyro.optim.Adam({"lr": 0.0001}),
#                      loss="ELBO")
#
# for i, (images, true_heights) in enumerate(data_generator(num_batches=10)):
#     print("loss is", svi.step(images))


# for epoch in range(2):
#     # initialize loss accumulator
#     epoch_loss = 0.
#     # do a training epoch over each mini-batch x
#     # returned by the data loader
#     for _, (x, _) in enumerate(train_loader):
#         # wrap the mini-batch in a PyTorch Variable
#         x = Variable(x)
#         # do ELBO gradient and accumulate loss
#         epoch_loss += svi.step(x)
#     print(epoch_loss)
