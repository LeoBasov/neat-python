#!/usr/bin/env python3

import random

from loc_module import LinInterpolNetwork as lin
from loc_module import InputNodeType as input_type
from loc_module import Mutator as mut

#Simulation parameters
NUMBER_NETWORKS = 100
NUMBER_ITTERATIONS = 100
DISCRITISATION = 7

#result values
NETWORKS = []
FITNESS_NETWOKR_PAIRS = []
MUTATOR = mut()

def main():
	print_header()
	print_set_up()

	set_up_networks()

	main_loop()

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

def print_footer():
	print(80*"-")
	print("Execution finished")
	print(80*"=")

def set_up_networks():
	for _ in range(NUMBER_NETWORKS):
		network = lin(DISCRITISATION)
		NETWORKS.append(network)

def main_loop():
	for i in range(NUMBER_ITTERATIONS):
		print("Evaluating network {}/{}".format(i + 1, NUMBER_ITTERATIONS), end="\r", flush=True)
		evaluate_networks()
		mutate()

def evaluate_networks():
	FITNESS_NETWOKR_PAIRS = []
	l_value = random.random()
	r_value = random.random()
	values = get_values(l_value, r_value)

	for network in NETWORKS:
		FITNESS_NETWOKR_PAIRS.append(evaluate_network(network, l_value, r_value, values))

	FITNESS_NETWOKR_PAIRS = sorted(FITNESS_NETWOKR_PAIRS, key = lambda x: x[0]) 
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
	input_values = ((l_value, input_type.L_VALUE.value), (r_value, input_type.R_VALUE.value))
	output_values = network.execute(input_values)
	fitness = 0

	for i in range(len(values)):
		fitness += abs(values[i] - output_values[input_type.MAX_ID.value + i])

	return [fitness/len(values), network]

def mutate():
	for i in range(len(FITNESS_NETWOKR_PAIRS)):
		if i < 0.2*len(NETWORKS):
			NETWORKS[i] = FITNESS_NETWOKR_PAIRS[i][1]
		else:
			NETWORKS[i] = MUTATOR.mutate(FITNESS_NETWOKR_PAIRS[i][1])

if __name__ == '__main__':
	main()