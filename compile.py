"""
Compile an artifact saved as a PersistentArtifact
"""
from artifact import PersistentArtifact
import argparse

parser = argparse.ArgumentParser("compile artifact")
parser.add_argument("name", help="Name of artifact top be compiled", nargs=1, type=str)
parser.add_argument("steps", help="Number of steps for compilation", nargs=1, type=int)
parser.add_argument("cuda", help="Whether to use GPU", nargs=1, type=int)

args = parser.parse_args()

print("CUDA:", bool(args.cuda[0]))
print("N_STEPS:", args.steps[0])

artifact = PersistentArtifact.load(args.name[0])
artifact.compile(args.steps[0], bool(args.cuda[0]))
