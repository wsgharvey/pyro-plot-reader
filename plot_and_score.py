"""
Calculates the log likelihood of data in a test file given the distributions predicted by an artifact
"""
from pyro.infer.csis.proposal_dists import UniformProposal
import torch
from torch.autograd import Variable

from file_paths import DATASET_FOLDER
from artifact import PersistentArtifact
import argparse

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
inference_log = artifact.infer(args.dataset, 
                               cuda=bool(args.cuda),
                               max_plots=args.N)

targets_file = open("{}/{}/test/targets.csv".format(DATASET_FOLDER, args.dataset), 'r')

num_incorrect_bar_counts = 0
num_correct_bar_counts = 0
log_pdf = 0

num_bars = 0

for trace in inference_log:
    true_values = targets_file.readline().strip('\n').split(',')

    true_num_bars = len(true_values)
    if "bar_height_{}".format(true_num_bars-1) in trace\
            and "bar_height_{}".format(true_num_bars) not in trace:

        for i, true_value in enumerate(map(float, true_values)):
            proposal_params = trace["bar_height_{}".format(i)]
            mode, certainty = proposal_params
            proposal_dist = UniformProposal(Variable(torch.Tensor([0.])),
                                            Variable(torch.Tensor([10.])),
                                            mode*10,
                                            certainty)
            log_pdf += proposal_dist.log_prob(Variable(torch.Tensor([true_value])))
            num_bars += 1
        num_correct_bar_counts += 1

    else:
        num_incorrect_bar_counts += 1

mean_log_pdf = log_pdf / num_bars

if args.L is not None:
    scores = pickle.load(open(args.L, 'rb'))
    scores.add_score(args.architecture, args.dataset, mean_log_pdf)
    pickle.dump(scores, open(args.L, 'wb'))
