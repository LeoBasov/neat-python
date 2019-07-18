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
		gene1 = Gene(in_node = 0, out_node = 4, weight = -2.32161229, enabled = True)
		gene2 = Gene(in_node = 0, out_node = 5, weight = -5.2368337, enabled = True)
		gene3 = Gene(in_node = 0, out_node = 3, weight = -3.13762134, enabled = True)

		gene4 = Gene(in_node = 1, out_node = 4, weight = 5.70223616, enabled = True)
		gene5 = Gene(in_node = 1, out_node = 5, weight = 3.42762429, enabled = True)

		gene6 = Gene(in_node = 2, out_node = 4, weight = 5.73141813, enabled = True)
		gene7 = Gene(in_node = 2, out_node = 5, weight = 3.4327536, enabled = True)

		gene8 = Gene(in_node = 4, out_node = 3, weight = 7.05553511, enabled = True)
		gene9 = Gene(in_node = 5, out_node = 3, weight = -7.68450564, enabled = True)

		self.set_genes([gene1, gene2, gene3, gene4, gene5, gene6, gene7, gene8, gene9])

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
		new_networks[i] = neat.mutate(evaluated_networks[i][1])
		new_networks[len(evaluated_networks) - 1 - i] = neat.mutate(evaluated_networks[i][1])

	return new_networks

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def main():
	number_networks = 1
	number_itterations = 1
	networks = generate_networks(number_networks)

	print("Evaluating networks")

	"""for i in range(number_itterations):
		print("Evaluating network {}/{}".format(i + 1, number_itterations), end="\r", flush=True)

		lis = evaluate(networks)
		networks = mutate(lis)"""

	print("")
	print_best(networks[0])
	print(80*"-")
	test_best(networks[0])
	print("done")

def print_best(networks):
	#print("Best fitness:", networks)

	for key, node in networks.nodes.items():
		print(node)

def test_best(network):
	val1 = 0
	val2 = 0

	_test_best(network, 0, 0)
	_test_best(network, 0, 1)
	_test_best(network, 1, 0)
	_test_best(network, 1, 1)	

def _test_best(network, val1, val2):
	input_values_node_ids = [(val1, 1), (val2, 2)]

	network.execute(input_values_node_ids)

	value = sigmoid(network.nodes[3].value)
	fitness = 1.0 - abs(float(val1 != val2) - sigmoid(network.nodes[3].value))

	print("Value 1: {}, Value 2: {}, XOR: {}, Network: {}, Fitness: {}".format(val1, val2, float(val1 != val2), value, fitness))

if __name__ == '__main__':
	main()