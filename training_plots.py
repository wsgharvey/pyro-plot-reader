"""
Make plots of the training loss for artifacts saved as PersistentArtifacts
"""
from artifact import PersistentArtifact
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("plot loss history of artifact during compilation")
parser.add_argument("output_file", help="Path to save plot to", type=str)
parser.add_argument("names", help="Names of artifact to be plotted", nargs='*', type=str)
parser.add_argument("-Y", help="Limit on highest y-axis value", type=float)
parser.add_argument("-P", help="Path of artifacts if not using default", type=str)
args = parser.parse_args()

fig, ax = plt.subplots()
for name in args.names:
    artifact = PersistentArtifact.load(name, artifact_folder=args.P)
    validation_losses = artifact.validation_losses
    x = [steps for steps, loss in validation_losses]
    y = [loss for steps, loss in validation_losses] 
    ax.plot(x, y, label=name)
    ax.set_ylim(top=args.Y)

plt.xlabel(r"Training Traces")
plt.ylabel(r"Validation Loss")
plt.legend()
plt.savefig(args.output_file)
