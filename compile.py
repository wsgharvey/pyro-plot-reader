"""
Compile an artifact saved as a PersistentArtifact
"""
from artifact import PersistentArtifact
import argparse

parser = argparse.ArgumentParser("compile artifact")
parser.add_argument("name", help="Name of artifact top be compiled", nargs=1, type=str)
parser.add_argument("steps", help="Number of steps for compilation", nargs=1, type=int)
parser.add_argument("cuda", help="Whether to use GPU", nargs=1, type=bool)

args = parser.parse_args()

artifact = PersistentArtifact.load(args.name)
artifact.compile(args.steps, args.cuda)
