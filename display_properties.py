"""
Display the properties of a PersistentArtifact
"""
from artifact import PersistentArtifact
import argparse

parser = argparse.ArgumentParser("display artifact properties")
parser.add_argument("name", help="Name of artifact top be compiled", nargs=1, type=str)

args = parser.parse_args()

artifact = PersistentArtifact.load(args.name[0])

print("guide_kwargs:\n", artifact.guide_kwargs)
print("compiler_kwargs:\n", artifact.compiler_kwargs)
print("optimiser_kwargs:\n", artifact.optimiser_kwargs)
print("number of validations:", len(artifact.validation_losses))