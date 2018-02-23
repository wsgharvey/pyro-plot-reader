from helpers import ScoreKeeper

import argparse
import pickle
import numpy as np

def print_spaced(word):
    word = str(word)
    print(word + ' '*(20-len(word)), end='')

LOSS_PATH = "/scratch/wsgh/plot-reader/losses/loss_data.p"

parser = argparse.ArgumentParser("make a table of previsouly calculated test scores")
parser.add_argument("-P", help="Path to file to save plots to", type=str)

args = parser.parse_args()

scores_object = pickle.load(open(LOSS_PATH, 'rb'))
scores = scores_object.scores

datasets = set(sum([list(a.keys()) for a in scores.values()], []))
architectures = set(scores.keys())

print_spaced('')
for dataset in datasets:
    print_spaced(dataset)
print()
for architecture in architectures:
    print_spaced(architecture)
    for dataset in datasets:
        if dataset in scores[architecture]:
            print_spaced(scores[architecture][dataset][-1].data.numpy()[0])
        else:
            print_spaced('')
    print()

