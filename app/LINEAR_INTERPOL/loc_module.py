import sys
import random
import math
import enum

sys.path.append('../../.')

from neat.genome import Genome
from neat.genome import Gene
from neat.neat import Mutator
from neat.neat import MutationType
from neat.neat import Probability

class LinInterpolGenome(Genome):
	def __init__(self, discretisation):
		super().__init__()
		genes = []
		hidden_nodes = []

		self.bias_node_id = 0
		
		self.input_node_id1 = self.add_input_node()
		self.input_node_id2 = self.add_input_node()

		for i in range(discretisation):
			hidden_nodes.append(self.add_hidden_node())

		for i in range(discretisation):
			self.add_output_node()

		for node_id in hidden_nodes:
			genes.append(Gene(self.bias_node_id, node_id, weight = self.__get_random_weight()))
			genes.append(Gene(self.input_node_id1, node_id, weight = self.__get_random_weight()))
			genes.append(Gene(self.input_node_id2, node_id, weight = self.__get_random_weight()))

		for output_node_id in self.output_nodes_ids:
			genes.append(Gene(self.bias_node_id, output_node_id, weight = self.__get_random_weight()))

			for hidden_node_id in hidden_nodes:
				genes.append(Gene(hidden_node_id, output_node_id, weight = self.__get_random_weight()))

		self.set_genes(genes)

	def __get_random_weight(self):
		return  10 - 20.0*random.random()

class Mutator(Mutator):
	def __init__(self):
		super().__init__()

		self.new_weight_range = 10.0
		self.weight_variation = 0.1
		self.max_network_size = 4

		self.probabilities = []

		self.probabilities.append(Probability(MutationType.NEW_CONNECTION, 0.0))
		self.probabilities.append(Probability(MutationType.NEW_NODE, 0.00))
		self.probabilities.append(Probability(MutationType.MODIFY_WEIGHT, 0.1))
		self.probabilities.append(Probability(MutationType.CHANGE_CONNECTION_STATUS, 0.0))
		self.probabilities.append(Probability(MutationType.NEW_WEIGHT, 0.01))

		self.probabilities.sort()