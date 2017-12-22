from file_paths import DATASET_PATH

sse = 0
n = 0
with open("{}/test/targets.csv".format(DATASET_PATH)) as targets,\
        open("{}/test_predictions.csv".format(DATASET_PATH)) as estimates:
    for target, estimate in zip(targets, estimates):
        n += 1
        error = float(target) - float(estimate)
        sse += error**2

rmse = (sse/n)**0.5

print("rmse:", rmse)
