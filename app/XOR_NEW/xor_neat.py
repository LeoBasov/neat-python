import sys
import random

sys.path.append('../../.')

from neat.neat import NEAT
from neat.genome import Gene
from neat.genome import Genome
from neat.network import Network
from neat.utility import sigmoid

class XOR_NEAT(NEAT):
	def __init__(self):
		super().__init__()

		self.input_node1 = 0
		self.input_node2 = 0

		self.output_node = 0

	def initiatlize(self, **kwargs):
		self.number_itterations = kwargs["number_itterations"]
		self.number_sub_cycles = kwargs["number_sub_cycles"]

		self.test_case_name = kwargs["test_case_name"]
		self.test_case_specifics = kwargs["test_case_specifics"]

		self.__initialze_networks(kwargs["number_networks"], kwargs["number_hidden_nodes"], kwargs["number_genes"])

	def evaluate_network(self, network):
		val1 = (0, 0)
		val2 = (0, 1)
		val3 = (1, 0)
		val4 = (1, 1)

		turple1 = self.exceute_network(val1[0], val1[1], network)
		turple2 = self.exceute_network(val2[0], val2[1], network)
		turple3 = self.exceute_network(val3[0], val3[1], network)
		turple4 = self.exceute_network(val4[0], val4[1], network)

		return (turple1[2] + turple2[2] + turple3[2] + turple4[2])/4.0

	def evaluate_best_network(self, network):
		val1 = (0, 0)
		val2 = (0, 1)
		val3 = (1, 0)
		val4 = (1, 1)

		turple1 = self.exceute_network(val1[0], val1[1], network)
		turple2 = self.exceute_network(val2[0], val2[1], network)
		turple3 = self.exceute_network(val3[0], val3[1], network)
		turple4 = self.exceute_network(val4[0], val4[1], network)

		return (turple1, turple2, turple3, turple4)

	def exceute_network(self, val1, val2, network):
		input_vals = ((val1, self.input_node1), (val2, self.input_node2))
		output_vals = network.execute(input_vals)

		return (float(val1 != val2), (output_vals[self.output_node]) ,(1.0 - abs(float(val1 != val2) - output_vals[self.output_node])))

	def __initialze_networks(self, number_networks, number_hidden_nodes, number_genes):
		for _ in range(number_networks):
			self.networks.append(self.__initialze_network(number_hidden_nodes, number_genes))

	def __initialze_network(self, number_hidden_nodes, number_genes):
		genome = Genome()
		genes = []

		genome.allocate_hidden_nodes(number_hidden_nodes)

		self.input_node1 = genome.add_input_node()
		self.input_node2 = genome.add_input_node()

		self.output_node = genome.add_output_node()

		genes.append(Gene(0               , self.output_node, 20 - 10*random.random()))
		genes.append(Gene(self.input_node1, self.output_node, 20 - 10*random.random()))
		genes.append(Gene(self.input_node2, self.output_node, 20 - 10*random.random()))

		genome.set_genes(genes)

		genome.allocate_genes(number_genes)

		return Network(genome)