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
parser.add_argument("architecture", help="Architecture of artifact being run (for adding score to repo)", type=str)
parser.add_argument("dataset", help="Name of dataset to use", type=str)
parser.add_argument("cuda", help="Whether to use GPU", type=int)
parser.add_argument("-N", help="Maximum number of plots to run inference on", type=int, default=np.inf)
parser.add_argument("-L", help="Path to file to save the loss to", type=str)

args = parser.parse_args()

print("CUDA:", bool(args.cuda))

artifact = PersistentArtifact.load(args.artifact)
log_pdf = artifact.infer(args.dataset,
                         attention_plots=True,
                         cuda=bool(args.cuda),
                         max_plots=args.N)

targets_file = open("{}/{}/test/targets.csv".format(DATASET_FOLDER, args.dataset), 'r')

print(log_pdf)

if args.L is not None:
    scores = pickle.load(open(args.L, 'rb'))
    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    scores.add_score(args.architecture, args.dataset, (args.artifact, mean_log_pdf, time_string))
    pickle.dump(scores, open(args.L, 'wb'))
