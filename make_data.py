from pyro.infer.csis.util import sample_from_prior

from file_paths import DATASET_PATH
from model import model

from PIL import Image
import numpy as np


def create_dataset(file_path,
                   n_data):
    targets = []

    for i in range(n_data):
        trace = sample_from_prior(model)
        returned = trace.nodes["_RETURN"]["value"]

        targets.append(",".join(map(str, returned["bar_heights"])))

        img = returned["image"]
        img = img.view(3, 200, 200)
        img = img.data.numpy()

        imgArray = np.zeros((200, 200, 3), 'uint8')
        imgArray[..., 0] = img[0]
        imgArray[..., 1] = img[1]
        imgArray[..., 2] = img[2]

        img = Image.fromarray(imgArray)
        img.save(file_path + "/graph_{}.png".format(i))

    with open(file_path + "/targets.csv", 'w') as f:
        f.write("\n".join(targets))


open(x, "{}/README.md".format(DATASET_PATH)).close()
create_dataset("{}/train".format(DATASET_PATH), 1000)
create_dataset("{}/validation".format(DATASET_PATH), 100)
create_dataset("{}/test".format(DATASET_PATH), 100)
