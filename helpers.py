import torch
from torch.autograd import Variable

from PIL import Image

import numpy as np


def fig2tensor(fig):
    """ Takes in a matplotlib figure and returns a torch float tensor of the
        given dimensions to represent the image
        Height and width depend on the matplotlib figure's dimensions
    """
    fig.canvas.draw()

    flat_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    width, height = fig.canvas.get_width_height()
    img = flat_array.reshape((height, width, 3))

    if img.shape != (200, 200, 3):
        img = Image.fromarray(img, 'RGB')
        img = img.resize((200, 200), Image.BILINEAR)
        img = np.array(img)

    img = img.transpose((2, 0, 1))  # put layer in the first dimension
    img = torch.from_numpy(img).type(torch.FloatTensor)

    return img


def pix2inches(inches, dpi=100):
    """ converts sizes in inches into pixels
        - takes in sequence of numbers
    """
    # just multiply by 100 and make it a tuple
    return tuple(map(lambda x: x/dpi,
                     inches))


def set_size_pixels(fig, size):
    """ returns figure with size set in pixels
        wraps matplotlib's set_size_inches
         - currently will act differently depending on monitor dpi
    """
    dpi = 100
    size = pix2inches(size, dpi=dpi)

    fig.set_size_inches(size)
    return fig


def image2variable(image):
    image = np.array(image.convert('RGB')).astype(np.float32)
    image = np.array([image[..., 0], image[..., 1], image[..., 2]])
    return Variable(torch.Tensor(image))


class ScoreKeeper(object):
    def __init__(self):
        self.scores = {}

    def add_score(self, architecture, dataset, entry):
        if architecture not in self.scores:
            self.scores[architecture] = {}
        if dataset not in self.scores[architecture]:
            self.scores[architecture][dataset] = []
        self.scores[architecture][dataset].append(entry)
