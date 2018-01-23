from scipy.stats import beta

import numpy as np

from helpers import set_size_pixels

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def beta_inv_cdf(p, a, b):
    dist = beta(a, b)
    upper = 1
    lower = 0
    for _ in range(50):
        guess = (upper + lower)/2
        if dist.cdf(guess) > p:
            upper = guess
        else:
            lower = guess
    return guess


# Calculate the confidence intervals and save them to a file
# with open('data/bar-3d/mode-and-certainties', 'r') as f, open('data/bar-3d/confidence-intervals', 'w') as g:
#     while True:
#         try:
#             mode, certainty = map(float, (f.readline()[:-2]).split(","))
#             norm_mode = mode/10
#             norm_cert = certainty+2
#             a = norm_mode * (norm_cert-2)
#             b = (1 - norm_mode) * (norm_cert-2)
#
#             bounds = map(lambda p: beta_inv_cdf(p, a, b)*10, [0.025, 0.975])
#
#             g.write("{},{}\n".format(*bounds))
#         except ValueError:
#             break


with open('data/bar-3d/test_predictions.csv', 'r') as pred, open('data/bar-3d/confidence-intervals', 'r') as conf:
    for graph_no in range(100):
        bar_heights = list(map(float, (pred.readline()[:-2]).split(",")))
        errors = np.zeros((2, 3))
        for i, height in enumerate(bar_heights):
            low, high = map(float, (conf.readline()[:-2]).split(","))
            errors[:, i] = [height-low, high-height]

        fig, ax = plt.subplots()
        ax.bar(range(3),
               bar_heights,
               label="Bar",
               yerr=errors,
               capsize=10)
        ax.set_ylim(0, 10)

        fig = set_size_pixels(fig, (200, 200))
        fig.savefig('/home/will/Documents/4yp/plots/plot-reader/predictions/checkpoint1/{}.pdf'.format(graph_no))
        plt.close()
