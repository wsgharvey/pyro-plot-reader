import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from helpers import set_size_pixels, fig2tensor


def conf_interval(samples, p=0.95):
    """
    returns a confidence interval (low, high) when given a list of discrete
    probabilities for samples [ (p, height), ... ]
    """
    p_low = (1-p)/2
    p_high = 1-p_low
    samples = sorted(samples, key=lambda x: x[1])

    cdf = 0
    h_low, h_high = None, None
    for p, h in samples:
        cdf += p
        if cdf > p_low and h_low is None:
            h_low = h
        if cdf > p_high:
            h_high = h
            return h_low, h_high
    print("fucking failed")


def plot_samples(samples, ground_truth, drawing='error_bars'):
    """
    samples is a list of tuples: (weight, bar_heights)
    bar_heights is 2D list (num_bar_charts x num_bars)

    ground_truth is a 2D list with same structure as bar_heights
    """
    samples = [sample for sample in samples if len(sample[1]) > 0]  # remove samples with num_bars_charts = 0
    samples = [sample for sample in samples if len(sample[1][0]) > 0]  # remove samples with num_bars = 0

    true_structure = (len(ground_truth), len(ground_truth[0]))
    max_height = max(max(map(max, ground_truth)), max(map(lambda sample: max(map(max, sample[1])), samples)))

    colours = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'][::-1]

    # sort samples into lists with same structure (and assign each structure a colour)
    max_weight = 0
    structures = {true_structure: []}
    structure_colours = {true_structure: 'tab:blue'}
    structure_probabilities = {true_structure: 0}
    for sample in samples:
        weight, bar_heights = sample
        structure = (len(bar_heights), len(bar_heights[0]))
        if structure not in structures:
            structures[structure] = []
            structure_probabilities[structure] = 0
            try:
                colour = colours.pop()
            except IndexError:
                colour = np.random.rand(3)
            structure_colours[structure] = colour
        structures[structure].append(sample)
        structure_probabilities[structure] += weight
        max_weight = max(weight, max_weight)
    max_structure_prob = max(p for p in structure_probabilities.values())

    # work out confidence intervals for each bar in each structure
    structure_conf_intervals = {}
    for structure, sampled_bar_heights in structures.items():
        structure_prob = structure_probabilities[structure]
        num_bar_charts, num_bars = structure

        if structure_prob > 0:
            sample_heights = [[[] for _ in range(num_bars)] for _ in range(num_bar_charts)]
            for weight, bar_heights in sampled_bar_heights:
                for bar_chart_num, bar_chart in enumerate(bar_heights):
                    for bar_num, bar_height in enumerate(bar_chart):
                        sample_heights[bar_chart_num][bar_num].append((weight/structure_prob, bar_height))
            confidence_intervals = [[conf_interval(weighted_bar_heights) for weighted_bar_heights in bar_chart] for bar_chart in sample_heights]
            low_intervals = [[weighted_bar_heights[0] for weighted_bar_heights in bar_chart] for bar_chart in confidence_intervals]
            high_intervals = [[weighted_bar_heights[1] for weighted_bar_heights in bar_chart] for bar_chart in confidence_intervals]
            structure_conf_intervals[structure] = [[low, high] for low, high in zip(low_intervals, high_intervals)]

    # plot everything
    image = 0
    for structure, sampled_bar_heights in structures.items():
        fig, ax = plt.subplots()
        structure_prob = structure_probabilities[structure]
        if structure_prob == 0:
            continue
        confidence_intervals = structure_conf_intervals[structure]
        colour = structure_colours[structure]
        num_bar_charts, num_bars = structure

        density = 0.5
        bar_width = 0.5/num_bar_charts

        for bar_chart_num in range(num_bar_charts):
            loose = bar_chart_num/(num_bar_charts)-0.5*(num_bar_charts-1)/num_bar_charts
            dense = bar_chart_num*bar_width - num_bar_charts*bar_width/2
            offset = density*dense + (1-density)*loose

            # plot the bar charts in white with 0 height just to get the x-axis
            ax.bar(np.arange(num_bars)+offset, np.zeros(num_bars), zorder=0, width=bar_width, color=(1, 1, 1), edgecolor=(1, 1, 1))
            if structure == true_structure:
                ax.bar(np.arange(num_bars)+offset, ground_truth[bar_chart_num], zorder=0, width=bar_width, color=(0.8, 0.8, 0.8), edgecolor=(0.8, 0.8, 0.8))

            if drawing == 'markers' or drawing == 'both':
                for weight, bar_heights in sampled_bar_heights:
                    bar_chart_heights = bar_heights[bar_chart_num]
                    ax.scatter(np.arange(num_bars)+offset, bar_chart_heights, zorder=1, color=colour, marker='x', alpha=weight/max_weight)

            if drawing == 'error_bars' or drawing == 'both':
                conf_int = confidence_intervals[bar_chart_num]
                low_bound = np.array(conf_int[0])
                diff = np.array(conf_int[1]) - low_bound
                ax.axes.errorbar(x=np.arange(num_bars)+offset, y=low_bound, yerr=[diff*0, diff], fmt='none', capsize=3, elinewidth=2, ecolor=colour, alpha=structure_prob/max_structure_prob)

        ax.set_ylim(0, max_height*1.1)
        plt.tick_params(                   # copied from https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
        fig = set_size_pixels(fig, (200, 200))

        fig.canvas.draw()
        image += np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((200, 200, 3))
        plt.clf()
    if image == 0:
        raise Exception("maybe all traces had -inf log weights")
    image = np.clip(image, 0, 255).astype('uint8')
    return Image.fromarray(image)

# ground_truth = [[3, 4, 3, 6], [2, 3.5, 3, 6.5]]
# samples = [(0.4, [[3.4, 4.2, 3.4, 5], [2.02, 3.1, 3.4, 6.1]]),
#            (0.3, [[2.4, 5, 2, 7], [1.8, 3.3, 3.2, 6]]),
#            (0.1, [[2.1, 3.4, 2.6, 5.5], [2.2, 3.6, 2.9, 6.4]]),
#            (0.2, [[3.5, 3.1, 6.8], [2.2, 3, 8]])]
# plot_samples(samples, ground_truth, error_bars=True)
