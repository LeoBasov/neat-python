import sys
import random
import copy

sys.path.append('../../.')

from neat.neat import NEAT
from neat.neat import Mutator
from neat.genome import Gene
from neat.genome import Genome
from neat.network import Network
from neat.utility import modified_sigmoid
from neat.utility import linear_interpol

class LIN_NEAT(NEAT):
	def __init__(self):
		super().__init__()

		self.input_node1 = 0
		self.input_node2 = 0

		self.output_nodes = []

	def evaluate_network(self, network):
		val_1 = random.random()
		val_2 = random.random()
		fitness = 0

		ret_vals = self.execute_network(val_1, val_2, network)

		for tuple in ret_vals:
			fitness += self.calc_fitness(tuple[0], tuple[1])

		return fitness/len(ret_vals)

	def calc_fitness(self, expected_val, calculated_val):
		diff = min(abs(expected_val - calculated_val), expected_val)

		return (expected_val - diff)/expected_val

	def execute_network(self, val_1, val_2, network):
		input_vals = ((val_1, self.input_node1), (val_2, self.input_node2))
		output_vals = network.execute(input_vals)
		ret_vals = [[0, 0] for _ in range(len(self.output_nodes))]
		step = 1.0/(len(self.output_nodes) + 1)

		for i in range(len(ret_vals)):
			ret_vals[i][0] = output_vals[self.output_nodes[i]]
			ret_vals[i][1] = linear_interpol(0.0, val_1, 1.0, val_2, (i + 1)*step)

		return ret_vals

	def evaluate_best_network(self, network):
		val_1 = random.random()
		val_2 = random.random()
		fitness = 0
		ret_tuples = []

		ret_vals = self.execute_network(val_1, val_2, network)

		for tuple in ret_vals:
			fitness = self.calc_fitness(tuple[0], tuple[1])
			ret_tuples.append((tuple[1], tuple[0], fitness))

		return ret_tuples

	def initialze_network(self, **kwargs):
		number_hidden_nodes = kwargs['number_hidden_nodes']
		number_genes = kwargs['number_genes']
		number_output_nodes = kwargs['number_output_nodes']
		new_weight_range = kwargs['new_weight_range']
		genome = Genome()
		genes = []

		genome.allocate_hidden_nodes(number_hidden_nodes)

		self.input_node1 = genome.add_input_node()
		self.input_node2 = genome.add_input_node()

		if not len(self.output_nodes):
			for _ in range(number_output_nodes):
				self.output_nodes.append(genome.add_output_node())

				genes.append(Gene(0               , self.output_nodes[-1], new_weight_range - 2*new_weight_range*random.random()))
				genes.append(Gene(self.input_node1, self.output_nodes[-1], new_weight_range - 2*new_weight_range*random.random()))
				genes.append(Gene(self.input_node2, self.output_nodes[-1], new_weight_range - 2*new_weight_range*random.random()))

		else:
			for i in range(number_output_nodes):
				genome.add_output_node()

				genes.append(Gene(0               , self.output_nodes[i], new_weight_range - 2*new_weight_range*random.random()))
				genes.append(Gene(self.input_node1, self.output_nodes[i], new_weight_range - 2*new_weight_range*random.random()))
				genes.append(Gene(self.input_node2, self.output_nodes[i], new_weight_range - 2*new_weight_range*random.random()))

		genome.set_genes(genes)

		genome.allocate_genes(number_genes)

		return Network(genome)
