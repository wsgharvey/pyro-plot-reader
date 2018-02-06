"""
Script to set up an artifact which can then be compiled etc. with other scripts
(with arguments set here)
"""
from artifact import PersistentArtifact
import argparse

parser = argparse.ArgumentParser("initialise artifact")
parser.add_argument("name", help="Descriptive name", nargs=1, type=str)
parser.add_argument("-m", help="Arguments for running the model", nargs='*', type=str)
parser.add_argument("-g", help="Arguments for initialising the guide", nargs='*', type=str)
parser.add_argument("-mg", help="Arguments shared between the model and guide", nargs='*', type=str)
parser.add_argument("-c", help="Arguments for compilation", nargs='*', type=str)
parser.add_argument("-o", help="Optimiser arguments", nargs='*', type=str)

args = parser.parse_args()

if args.g is None:
    args.g = []
if args.c is None:
    args.c = []
if args.o is None:
    args.o = []

guide_kwargs = {k: eval(v) for k, v in map(lambda x: x.split("="), args.g)}
compiler_kwargs = {k: eval(v) for k, v in map(lambda x: x.split("="), args.c)}
optimiser_kwargs = {k: eval(v) for k, v in map(lambda x: x.split("="), args.o)}

a = PersistentArtifact(args.name[0],
                       guide_kwargs=guide_kwargs,
                       compiler_kwargs=compiler_kwargs,
                       optimiser_kwargs=optimiser_kwargs)
# Saves automatically
