import csv
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, dir = ''):
        self.dir = dir

    def plot_fitness(self, file_name):
        percentiles = []
        percentiles_fin = [[] for _ in range(10)]
        mean_vals = []

        with open(self.dir + '/' + file_name) as file:
            reader = csv.reader(file)

            for row in reader:
                fitness =  np.array([ float(i) for i in row ])
                percentiles.append(self.chunk_it(fitness, 10))

                mean_vals.append(np.mean(fitness))

        for step in percentiles:
            for i in range(len(percentiles_fin)):
                percentiles_fin[i].append(step[i])

        self._plot_fitness(percentiles_fin, mean_vals)

    def _plot_fitness(self, percentiles, mean_vals):
        plt.plot(mean_vals, label='Mean Fitness', color='black')

        plt.plot(percentiles[0], label='1 Percentile', linestyle = '--')
        plt.plot(percentiles[1], label='2 Percentile', linestyle = '--')
        plt.plot(percentiles[2], label='3 Percentile', linestyle = '--')
        plt.plot(percentiles[3], label='4 Percentile', linestyle = '--')
        plt.plot(percentiles[4], label='5 Percentile', linestyle = '--')
        plt.plot(percentiles[5], label='6 Percentile', linestyle = '--')
        plt.plot(percentiles[6], label='7 Percentile', linestyle = '--')
        plt.plot(percentiles[7], label='8 Percentile', linestyle = '--')
        plt.plot(percentiles[8], label='9 Percentile', linestyle = '--')
        plt.plot(percentiles[9], label='10 Percentile', linestyle = '--')

        plt.xlabel('x label')
        plt.ylabel('y label')

        plt.title("Simple Plot")

        plt.legend()

        plt.show()

    def chunk_it(self, seq, num):
        avg = len(seq) / float(num)
        inter = []
        out = []
        last = 0.0

        while last < len(seq):
            inter.append(seq[int(last):int(last + avg)])
            last += avg

        for iner_fit in inter:
            out.append(np.mean(iner_fit))

        return out
