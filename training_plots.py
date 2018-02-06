"""
Make plots of the training loss for artifacts saved as PersistentArtifacts
"""
from artifact import PersistentArtifact
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("compile artifact")
parser.add_argument("output_file", help="Path to save plot to", type=str)
parser.add_argument("names", help="Names of artifact to be plotted", nargs='*', type=str)

args = parser.parse_args()

for name in args.names:
    artifact = PersistentArtifact.load(name)
    validation_losses = artifact.validation_losses
    x = [steps for steps, loss in validation_losses]
    y = [loss for steps, loss in validation_losses]
    plt.plot(x, y, label=name)

plt.xlabel(r"Training Traces")
plt.ylabel(r"Validation Loss")
plt.legend()
plt.savefig(args.output_file)
