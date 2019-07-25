#!/usr/bin/env python3

import random
import copy
import sys

sys.path.append('../../.')

from loc_module import LinInterpolGenome as gen
from loc_module import Mutator as mut
from neat.network_slim import Network
from neat.genome import Genome

#Simulation parameters
NUMBER_NETWORKS = 100
NUMBER_ITTERATIONS = 100
NUMBER_SUB_CYCLES = 100

DISCRITISATION = 5

#result values
NETWORKS = []
FITNESS_NETWOKR_PAIRS = []
MUTATOR = mut()

def main():
	print_header()
	print_set_up()

	set_up_networks()

	main_loop()

	print_best()
	print_footer()

def print_header():
	print(80*"=")
	print("NEAT - NeuroEvolution of Augmenting Topologies")
	print("(c) 2019, Leo Basov")
	print(80*"-")
	print("LINEAR INTERPOLATION TEST CASE")
	print("Interpolation length: 1")
	print(80*"-")

def print_set_up():
	print("NUMBER_NETWORKS", NUMBER_NETWORKS)
	print("NUMBER_ITTERATIONS", NUMBER_ITTERATIONS)
	print("NUMBER_SUB_CYCLES", NUMBER_SUB_CYCLES)
	print("NUMBER_EVALUATED_POINTS", DISCRITISATION)
	print(80*"-")

def print_footer():
	print(80*"-")
	print("Execution finished")
	print(80*"=")

def print_best():
	print("FITNESS OF BEST NETWORK:", FITNESS_NETWOKR_PAIRS[0][0])
	print(80*"-")

	fitness_real_calc = evaluate_best_network(FITNESS_NETWOKR_PAIRS[0][1])

	for vals in fitness_real_calc:
		print("EXPECTED VALUE: {} CALCULATED VALUE: {} FITNESS: {}".format(round(vals[1], 3), round(vals[2], 3), round(vals[0], 3)))

	print(80*"-")

	for node in FITNESS_NETWOKR_PAIRS[0][1].nodes:
		print(node)

def set_up_networks():
	for _ in range(NUMBER_NETWORKS):
		genome = gen(DISCRITISATION)
		network = Network(genome)
		NETWORKS.append(network)

def main_loop():
	for i in range(NUMBER_ITTERATIONS):
		mean_fitness = 0
		Genome.LAST_ITTERATION = []

		for pair in FITNESS_NETWOKR_PAIRS:
			mean_fitness += pair[0]

		if len(FITNESS_NETWOKR_PAIRS):
			mean_fitness /= len(FITNESS_NETWOKR_PAIRS)

		print("MEAN FITNESS: %0.3f EVALUATED NETWORKS %0.0d/%0.0d" % (round(mean_fitness,3), i + 1, NUMBER_ITTERATIONS), end="\r", flush=True)
		evaluate_networks()
		mutate()

	evaluate_networks()
	print("")
	print(80*"-")

def evaluate_networks():
	FITNESS_NETWOKR_PAIRS.clear()
	l_value = random.random()
	r_value = random.random()
	values = get_values(l_value, r_value)

	for network in NETWORKS:
		fitness_values = []

		for i in range(NUMBER_SUB_CYCLES):
			fitness_values.append(evaluate_network(network, l_value, r_value, values))


		FITNESS_NETWOKR_PAIRS.append((sum(fitness_values)/NUMBER_SUB_CYCLES, network))

	FITNESS_NETWOKR_PAIRS.sort(key = lambda x: x[0])
	FITNESS_NETWOKR_PAIRS.reverse()

def get_values(l_value, r_value):
	values = []
	step = (r_value - l_value)/(DISCRITISATION + 1)

	for i in range(DISCRITISATION):
		x = l_value + step*(i + 1)
		value = l_value*(1.0 - (x - l_value)/(r_value - l_value)) + r_value*((x - l_value)/(r_value - l_value))

		values.append(value)

	return values

def evaluate_network(network, l_value, r_value, values):
	input_values = ((l_value, 1), (r_value, 2))
	output_values = network.execute(input_values)
	fitness = 0

	for i in range(len(values)):
		fitness += calc_fitness(values[i], output_values[3 + DISCRITISATION + i])

	return fitness/len(values)

def evaluate_best_network(network):
	l_value = random.random()
	r_value = random.random()
	values = get_values(l_value, r_value)
	input_values = ((l_value, 1), (r_value, 2))
	output_values = network.execute(input_values)
	fitness_real_calc = []

	for i in range(len(values)):
		real_value = values[i]
		calc_value = output_values[3 + DISCRITISATION + i]
		fitness = calc_fitness(real_value, calc_value)

		fitness_real_calc.append((fitness, real_value, calc_value))

	return fitness_real_calc

def calc_fitness(expected, calculated):
	return 1.0 - abs(expected - calculated)/max(expected, calculated)

def mutate():
	for i in range(len(FITNESS_NETWOKR_PAIRS)):
		if (i < 0.2*len(FITNESS_NETWOKR_PAIRS)) and (FITNESS_NETWOKR_PAIRS[i][0] > 0.75):
			NETWORKS[i] = FITNESS_NETWOKR_PAIRS[i][1]
		else:
			NETWORKS[i] = MUTATOR.mutate(FITNESS_NETWOKR_PAIRS[i][1])

if __name__ == '__main__':
	main()