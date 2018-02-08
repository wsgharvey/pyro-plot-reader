from pyro.infer.csis.util import sample_from_prior

from file_paths import DATASET_FOLDER
from model import Model

import os

from PIL import Image
import numpy as np
import torch

import argparse
import pickle

from artifact import PersistentArtifact

parser = argparse.ArgumentParser("Create a dataset")
parser.add_argument("name", help="Name of the dataset", type=str)
parser.add_argument("-a", help="An artifact to copy the keyword arguments from. If not provided, will use defaults", type=str)
args = parser.parse_args()
if args.a is not None:
    artifact = PersistentArtifact.load(args.a)
    model_kwargs = artifact.model_kwargs
    del artifact
else:
    model_kwargs = {}
model = Model(**model_kwargs)


def create_dataset(file_path,
                   n_data):
    targets = []
    for i in range(n_data):
        trace = sample_from_prior(model)
        returned = trace.nodes["_RETURN"]["value"]

        targets.append(",".join(map(str, returned["bar_heights"])))

        img = returned["image"]
        img = img.view(3, 210, 210)
        img = img.data.numpy()

        imgArray = np.zeros((210, 210, 3), 'uint8')
        imgArray[..., 0] = img[0]
        imgArray[..., 1] = img[1]
        imgArray[..., 2] = img[2]

        img = Image.fromarray(imgArray)
        img.save(file_path + "/graph_{}.png".format(i))

    with open(file_path + "/targets.csv", 'w') as f:
        f.write("\n".join(targets))


dataset_path = "{}/{}".format(DATASET_FOLDER, args.name)
if os.path.exists(dataset_path):
    raise Exception("There's already a folder at {}".format(dataset_path))
os.makedirs(dataset_path)

# Create README and directories
open("{}/README.md".format(dataset_path), 'a').close()
os.makedirs("{}/train".format(dataset_path))
os.makedirs("{}/validation".format(dataset_path))
os.makedirs("{}/test".format(dataset_path))
pickle.dump(model_kwargs, open("{}/data_params.p".format(dataset_path), 'wb'))

# Fill with data
torch.manual_seed(0)
create_dataset("{}/train".format(dataset_path), 1000)
create_dataset("{}/validation".format(dataset_path), 100)
create_dataset("{}/test".format(dataset_path), 100)
