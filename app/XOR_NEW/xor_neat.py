import sys
import random

sys.path.append('../../.')

from neat.neat import NEAT

from neat.genome import Gene
from neat.genome import Genome

from neat.network import Network

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

		self.__initialze_networks(kwargs["number_networks"])

	def evaluate_network(self, network):
		return 0

	def evaluate_best_network(self, network):
		return [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

	def __initialze_networks(self, number_networks):
		for _ in range(number_networks):
			self.networks.append(self.__initialze_network())

	def __initialze_network(self):
		genome = Genome()
		genes = []

		self.input_node1 = genome.add_input_node()
		self.input_node2 = genome.add_input_node()

		self.output_node = genome.add_output_node()

		genes.append(Gene(0               , self.output_node, 20 - 10*random.random()))
		genes.append(Gene(self.input_node1, self.output_node, 20 - 10*random.random()))
		genes.append(Gene(self.input_node2, self.output_node, 20 - 10*random.random()))

		genome.set_genes(genes)

		return Network(genome)

	def mutate(self):
		pass

