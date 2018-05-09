"""
Calculates the log likelihood of data in a test file given the distributions predicted by an artifact
"""
from pyro.infer.csis.proposal_dists import UniformProposal
import torch
from torch.autograd import Variable

from file_paths import DATASET_FOLDER
from artifact import PersistentArtifact
import argparse

import datetime
import pickle
import numpy as np

from helpers import ScoreKeeper


parser = argparse.ArgumentParser("run inference with artifact and plot results")
parser.add_argument("artifact", help="Name of artifact to run", type=str)
parser.add_argument("dataset", help="Name of dataset to use", type=str)
parser.add_argument("cuda", help="Whether to use GPU", type=int)
parser.add_argument("graph_no", help="Index of graph to run inference on", type=int, default=0)
parser.add_argument("-N", help="Number of plot inference traces to use", type=int, default=10)

args = parser.parse_args()

print("CUDA:", bool(args.cuda))

artifact = PersistentArtifact.load(args.artifact)

artifact.posterior_samples(args.dataset,
                           max_plots=1,
                           start_no=args.graph_no,
                           n_traces=args.N,
                           cuda=bool(args.cuda))
