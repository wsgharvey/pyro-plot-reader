"""
Script to set up an artifact which can then be compiled etc. with other scripts
(with arguments set here)
"""
from artifact import PersistentArtifact
import argparse

parser = argparse.ArgumentParser("initialise artifact")
parser.add_argument("name", help="Descriptive name", nargs=1, type=str)
parser.add_argument("-copy", help="Artifact to copy arguments from, replacing only the ones specified", nargs=1, type=str)
parser.add_argument("-m", help="Arguments for running the model", nargs='*', type=str)
parser.add_argument("-g", help="Arguments for initialising the guide", nargs='*', type=str)
parser.add_argument("-mg", help="Arguments shared between the model and guide", nargs='*', type=str)
parser.add_argument("-c", help="Arguments for compilation", nargs='*', type=str)
parser.add_argument("-o", help="Optimiser arguments", nargs='*', type=str)

args = parser.parse_args()

if args.m is None:
    args.m = []
if args.g is None:
    args.g = []
if args.mg is None:
    args.mg = []
if args.c is None:
    args.c = []
if args.o is None:
    args.o = []

if args.copy is not None:
    mould = PersistentArtifact.load(args.copy[0])
    model_kwargs = mould.model_kwargs
    guide_kwargs = mould.guide_kwargs
    compiler_kwargs = mould.compiler_kwargs
    optimiser_kwargs = mould.optimiser_kwargs
else:
    model_kwargs = {}
    guide_kwargs = {}
    compiler_kwargs = {}
    optimiser_kwargs = {}

def string_eval(v):
    try:
        return eval(v)
    except:
        return v

model_kwarg_changes = {k: string_eval(v) for k, v in map(lambda x: x.split("="), args.m)}
guide_kwarg_changes = {k: string_eval(v) for k, v in map(lambda x: x.split("="), args.g)}
model_and_guide_kwarg_changes = {k: string_eval(v) for k, v in map(lambda x: x.split("="), args.mg)}
compiler_kwarg_changes = {k: string_eval(v) for k, v in map(lambda x: x.split("="), args.c)}
optimiser_kwarg_changes = {k: string_eval(v) for k, v in map(lambda x: x.split("="), args.o)}

model_kwargs.update(model_kwarg_changes)
guide_kwargs.update(guide_kwarg_changes)
model_kwargs.update(model_and_guide_kwarg_changes)
guide_kwargs.update(model_and_guide_kwarg_changes)
compiler_kwargs.update(compiler_kwarg_changes)
optimiser_kwargs.update(optimiser_kwarg_changes)

a = PersistentArtifact(args.name[0],
                       model_kwargs=model_kwargs,
                       guide_kwargs=guide_kwargs, 
                       compiler_kwargs=compiler_kwargs,
                       optimiser_kwargs=optimiser_kwargs)
# Saves automatically
