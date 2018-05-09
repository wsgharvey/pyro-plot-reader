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


parser = argparse.ArgumentParser("gives log prob of test set according to model")
parser.add_argument("artifact", help="Name of artifact to run (can be any)", type=str)
parser.add_argument("dataset", help="Name of dataset to use", type=str)
parser.add_argument("-N", help="Maximum number of plots to run inference on", type=int, default=100)

args = parser.parse_args()

artifact = PersistentArtifact.load(args.artifact)

log_pdf = 0
failed = 0
for start_no in range(0, args.N, 1):
    print(start_no)
    log_pdf += artifact.model_log_prob(args.dataset,
                                       attention_plots=True,
                                       start_no=start_no,
                                       max_plots=1)

print(log_pdf)
print("succeeded on", args.N-failed, "out of", args.N)
