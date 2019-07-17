#!/usr/bin/env python3

import sys
import random
import math

sys.path.append('../../.')

from neat.neat import Node
from neat.neat import Gene
from neat.neat import Network
from neat.neat import NEAT

class XORNetwork(Network):
	def __init__(self):
		super().__init__()

		self._add_input_node(1)
		self._add_input_node(2)

		self._add_output_node(3)

		self._set_up_genes()

	def _set_up_genes(self):
		gene1 = Gene(in_node = 0, out_node = 3, weight = 1.0, enabled = True)

		self.set_genes([gene1])

def generate_networks(number):
	networks = []

	for i in range(number):
		network = XORNetwork()
		networks.append(network)

	return networks

def evaluate(networks):
	lis = []

	for network in networks:
		val1 = round(random.random())
		val2 = round(random.random())

		input_values_node_ids = [(val1, 1), (val2, 2)]

		network.execute(input_values_node_ids)

		fitness = 1.0 - abs(float(val1 != val2) - sigmoid(network.nodes[3].value))

		lis.append((fitness, network))

	lis = sorted(lis, key = lambda x: x[0]) 
	lis.reverse()

	return lis

def mutate(evaluated_networks):
	neat = NEAT()
	new_networks = len(evaluated_networks)*[None]

	for i in range(len(evaluated_networks)):
		if evaluated_networks[i][0] < 0.8:
			new_networks[i] = neat.mutate(evaluated_networks[i][1])
		else:
			new_networks[i] = evaluated_networks[i][1]

	return new_networks

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def main():
	number_networks = 1
	number_itterations = 100
	networks = generate_networks(number_networks)

	print("Evaluating networks")

	for i in range(number_itterations):
		print("Evaluating network {}/{}".format(i + 1, number_itterations), end="\r", flush=True)

		lis = evaluate(networks)
		networks = mutate(lis)

	print("")
	print_best(lis)
	print("done")

def print_best(lis):
	print("Best fitness:", lis[0][0])

	for key, node in lis[0][1].nodes.items():
		print(node)

if __name__ == '__main__':
	main()