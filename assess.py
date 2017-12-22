import numpy as np

from file_paths import DATASET_PATH

sse = 0
n = 0
with open("{}/test/targets.csv".format(DATASET_PATH)) as targets,\
        open("{}/test_predictions.csv".format(DATASET_PATH)) as estimates:
    for target, estimate in zip(targets, estimates):
        n += 1
        target_vec = np.array(list(map(float, target.split(","))))
        estimate_vec = np.array(list(map(float, estimate.split(","))))
        diff = target_vec - estimate_vec
        sse += sum(diff**2)/len(target_vec)

rmse = (sse/n)**0.5

print("rmse:", rmse)
