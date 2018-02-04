from PIL import Image


class AttentionTracker(object):
    def __init__(self, folder_path):
        self.n_passes = 0
        self.graphics = []
        self.folder_path = folder_path

    def add_graphic(self, array):
        """
        Adds an image for where is paid attention to at one particular stage
        given in form of 2D numpy array
        """
        array = array.astype('uint8')
        img = Image.fromarray(array)
        img = img.convert('RGB')
        self.graphics.append(img)

    def save_graphics(self):
        """
        Saves the graphics so far to given location and resets
        """
        for i, graphic in enumerate(self.graphics):
            graphic.save("{}/{}-{}.png".format(self.folder_path,
                                               self.n_passes,
                                               i))
        self.n_passes += 1
        self.graphics = []
