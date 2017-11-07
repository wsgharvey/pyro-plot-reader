import torch
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
