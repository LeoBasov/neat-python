#!/usr/bin/env python3

import sys
import random
import math

sys.path.append('../../.')

from neat.genome import Genome
from neat.genome import Gene
from neat.network import Network
from neat.neat import Mutator

class XORGenome(Genome):
	def __init__(self):
		super().__init__()
		genes= []

		self.input_node_id1 = self.add_input_node()
		self.input_node_id2 = self.add_input_node()

		self.output_node_id1 = self.add_output_node()

		genes.append(Gene(0,                   self.output_node_id1, weight = 10 - 20.0*random.random()))
		genes.append(Gene(self.input_node_id1, self.output_node_id1, weight = 10 - 20.0*random.random()))
		genes.append(Gene(self.input_node_id2, self.output_node_id1, weight = 10 - 20.0*random.random()))

		self.set_genes(genes)

def generate_networks(number):
	networks = []

	for i in range(number):
		network = Network(XORGenome())
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
	neat = Mutator()
	rest = int(0.5*len(evaluated_networks))
	new_networks = len(evaluated_networks)*[None]

	for i in range(len(evaluated_networks)):
		mate1 = random.choice(evaluated_networks[:rest])[1]
		mate2 = random.choice(evaluated_networks[:rest])[1]

		while mate1 == mate2:
			mate1 = random.choice(evaluated_networks[:rest])[1]
			mate2 = random.choice(evaluated_networks[:rest])[1]

		total_list = Genome.set_up_lists(mate1.genome, mate2.genome)
		nodes = mate1.genome.nodes if len(mate1.genome.nodes) > len(mate2.genome.nodes) else mate2.genome.nodes
		genome = Genome.mate(total_list, nodes)
		network = Network(genome)

		new_networks[i] = neat.mutate(network)

	return new_networks

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def main():
	number_networks = 150
	number_itterations = 50
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

	for fitness, network in lis:
		print("FITNESS: {} | NODE NUMBER: {} | GENE NUMBER: {}".format(fitness, len(network.nodes), len(network.genome.genes)))

	print("done")

def print_best(lis):
	print("Best fitness:", lis[0][0])

	for node in lis[0][1].nodes:
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