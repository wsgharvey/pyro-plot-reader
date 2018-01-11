from pyro.infer.csis.util import sample_from_prior

from file_paths import DATASET_PATH
from model import model

import os

from PIL import Image
import numpy as np


def create_dataset(file_path,
                   n_data):
    targets = []
    for i in range(n_data):
        trace = sample_from_prior(model)
        # messy_image = trace.nodes["observed_image"]["value"].view(3, 200, 200)
        # messy_image = F.relu(messy_image - F.relu(messy_image-255))
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

# Create README and directories if they don't already exist
open("{}/README.md".format(DATASET_PATH), 'a').close()
if not os.path.exists("{}/train".format(DATASET_PATH)):
    os.makedirs("{}/train".format(DATASET_PATH))
if not os.path.exists("{}/validation".format(DATASET_PATH)):
    os.makedirs("{}/validation".format(DATASET_PATH))
if not os.path.exists("{}/test".format(DATASET_PATH)):
    os.makedirs("{}/test".format(DATASET_PATH))

# Fill with data
create_dataset("{}/train".format(DATASET_PATH), 1000)
create_dataset("{}/validation".format(DATASET_PATH), 100)
create_dataset("{}/test".format(DATASET_PATH), 100)
