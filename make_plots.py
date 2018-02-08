"""
Run infrence and generate plots with an artifact saved as a PersistentArtifact
"""
from artifact import PersistentArtifact
import argparse

import numpy as np

parser = argparse.ArgumentParser("run inference with artifact and plot results")
parser.add_argument("artifact", help="Name of artifact to run", type=str)
parser.add_argument("dataset", help="Name of dataset to use", type=str)
parser.add_argument("cuda", help="Whether to use GPU", type=int)
parser.add_argument("-N", help="Maximum number of plots to run inference on", type=int, default=np.inf)

args = parser.parse_args()

print("CUDA:", bool(args.cuda))

artifact = PersistentArtifact.load(args.artifact)
artifact.make_plots(args.dataset, cuda=bool(args.cuda), max_plots=args.N)
