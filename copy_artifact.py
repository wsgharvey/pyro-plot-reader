"""
Copy an artifact saved as a PersistentArtifact
"""
from artifact import PersistentArtifact
import argparse

parser = argparse.ArgumentParser("compile artifact")
parser.add_argument("name", help="Name of artifact to be copied", nargs=1, type=str)
parser.add_argument("new_name", help="Name to copy the artifact to", nargs=1, type=str)

args = parser.parse_args()

artifact = PersistentArtifact.load(args.name[0])
artifact.copy(args.new_name[0])
