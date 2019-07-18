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

		self._add_hidden_node(4)

		self._add_output_node(3)

		self._set_up_genes()

	def _set_up_genes(self):
		gene1 = Gene(in_node = 0, out_node = 3, weight = 10 - 20.0*random.random(), enabled = True)
		gene2 = Gene(in_node = 1, out_node = 3, weight = 10 - 20.0*random.random(), enabled = True)
		gene3 = Gene(in_node = 2, out_node = 3, weight = 10 - 20.0*random.random(), enabled = True)

		self.set_genes([gene1, gene2, gene3])

def generate_networks(number):
	networks = []

	for i in range(number):
		network = XORNetwork()
		networks.append(network)

	return networks

def evaluate(networks):
	lis = []

	for network in networks:
		fitness = 0

		val11 = 0
		val21 = 0

		val12 = 0
		val22 = 1

		val13 = 1
		val23 = 0

		val14 = 1
		val24 = 1

		fitness += execute(val11, val21, network)
		fitness += execute(val12, val22, network)
		fitness += execute(val13, val23, network)
		fitness += execute(val14, val24, network)

		lis.append((fitness/4, network))

	lis = sorted(lis, key = lambda x: x[0]) 
	lis.reverse()

	return lis

def execute(val1, val2, network):
	input_values_node_ids = [(val1, 1), (val2, 2)]

	network.execute(input_values_node_ids)
	val = 1.0 - abs(float(val1 != val2) - sigmoid(network.nodes[3].value))

	return val

def mutate(evaluated_networks):
	neat = NEAT()
	new_networks = len(evaluated_networks)*[None]

	for i in range(len(evaluated_networks)):
		if (i < 0.2*len(evaluated_networks)) and (evaluated_networks[i][0] > 0.75):
			new_networks[i] = evaluated_networks[i][1]
		elif i < 0.6*len(evaluated_networks):
			new_networks[i] = neat.mutate(evaluated_networks[i][1])
		else:
			genes = evaluated_networks[i][1].genes

			for gene in evaluated_networks[i][1].genes:
				gene.weight = 10 - 20.0*random.random()

			evaluated_networks[i][1].set_genes(genes)

			new_networks[i] = evaluated_networks[i][1]

	return new_networks

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def main():
	number_networks = 100
	number_itterations = 10000
	networks = generate_networks(number_networks)

	print("Evaluating networks")

	for i in range(number_itterations):
		print("Evaluating network {}/{}".format(i + 1, number_itterations), end="\r", flush=True)

		lis = evaluate(networks)
		networks = mutate(lis)

	lis = evaluate(networks)

	print("")
	print_best(lis)
	print(80*"-")
	test_best(lis[0][1])
	print("done")

def print_best(lis):
	print("Best fitness:", lis[0][0])

	for key, node in lis[0][1].nodes.items():
		print(node)

	"""print(80*"-")

	print("Second best fitness:", lis[1][0])

	for key, node in lis[1][1].nodes.items():
		print(node)

	print(80*"-")

	print("Third best fitness:", lis[2][0])

	for key, node in lis[2][1].nodes.items():
		print(node)

	print(80*"-")

	print("Tenth:", lis[10][0])

	for key, node in lis[10][1].nodes.items():
		print(node)

	print(80*"-")"""

def test_best(network):
	val11 = 0
	val21 = 0

	val12 = 0
	val22 = 1

	val13 = 1
	val23 = 0

	val14 = 1
	val24 = 1

	input_values_node_ids = [(val11, 1), (val21, 2)]

	network.execute(input_values_node_ids)

	value = sigmoid(network.nodes[3].value)
	fitness = 1.0 - abs(float(val11 != val21) - sigmoid(network.nodes[3].value))

	print("Value 1: {}, Value 2: {}, XOR: {}, Network: {}, Fitness: {}".format(val11, val21, int(val11 != val21), round(value), fitness))

	input_values_node_ids = [(val12, 1), (val22, 2)]

	network.execute(input_values_node_ids)

	value = sigmoid(network.nodes[3].value)
	fitness = 1.0 - abs(float(val12 != val22) - sigmoid(network.nodes[3].value))

	print("Value 1: {}, Value 2: {}, XOR: {}, Network: {}, Fitness: {}".format(val12, val22, int(val12 != val22), round(value), fitness))

	input_values_node_ids = [(val13, 1), (val23, 2)]

	network.execute(input_values_node_ids)

	value = sigmoid(network.nodes[3].value)
	fitness = 1.0 - abs(float(val13 != val23) - sigmoid(network.nodes[3].value))

	print("Value 1: {}, Value 2: {}, XOR: {}, Network: {}, Fitness: {}".format(val13, val23, int(val13 != val23), round(value), fitness))

	input_values_node_ids = [(val14, 1), (val24, 2)]

	network.execute(input_values_node_ids)

	value = sigmoid(network.nodes[3].value)
	fitness = 1.0 - abs(float(val14 != val24) - sigmoid(network.nodes[3].value))

	print("Value 1: {}, Value 2: {}, XOR: {}, Network: {}, Fitness: {}".format(val14, val24, int(val14 != val24), round(value), fitness))

if __name__ == '__main__':
	main()