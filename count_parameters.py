"""
Count the number of parameters used in an artifact saved as a PersistentArtifact
"""
from artifact import PersistentArtifact
from guide import Guide

import argparse
import numpy as np

parser = argparse.ArgumentParser("counts trainable parameters of a specified artifact")
parser.add_argument("name", help="Name of artifact to have parameters counted", nargs=1, type=str)

args = parser.parse_args()

artifact = PersistentArtifact.load(args.name[0])
guide_kwargs = artifact.guide_kwargs.copy()
guide = Guide(**guide_kwargs)

guide_parameters = filter(lambda p: p.requires_grad, guide.parameters())
params = sum([np.prod(p.size()) for p in guide_parameters])

print(params)
