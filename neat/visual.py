import csv
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

class Visualizer:
    def __init__(self, dir = ''):
        self.dir = dir

    def plot_species(self, fig_name):
        path = self.dir + '/species'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        species = []

        for file in files:
            loc_species = [[], []]

            with open(path + '/' + file, 'r') as file:
                reader = csv.reader(file)

                for row in reader:
                    loc_species[0].append(int(row[0]))
                    loc_species[1].append(float(row[1]))

            species.append((loc_species, file.name.split('_')[2].split('.')[0]))

        self._plot_species(species, self.dir + '/' + fig_name)

    def _plot_species(self, species, fig_name):
        for spec in species:
            plt.plot(spec[0][0], spec[0][1], label = spec[1])

        plt.xlabel('Generation [-]')
        plt.ylabel('Size [-]')

        plt.title("Species plot")
        #plt.legend()

        plt.savefig(fig_name, dpi = 600)

        plt.show()

    def plot_fitness(self, file_name, fig_name):
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

        self._plot_fitness(percentiles_fin, mean_vals, self.dir + '/' + fig_name)

    def _plot_fitness(self, percentiles, mean_vals, fig_name):
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

        plt.xlabel('Generation [-]')
        plt.ylabel('Fitness [-]')

        plt.title("Fitness plot")

        #plt.legend()

        plt.savefig(fig_name, dpi = 600)

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
