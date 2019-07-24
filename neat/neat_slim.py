"""NEAT
Copyright (C) 2019  Leo Basov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
long with this program. If not, see <https://www.gnu.org/licenses/>."""

import copy
import random

from .network_slime import Node
from .network_slime import Network
from .network_slime import Gene
from .network_slime import Genenome


class NEAT:
	def __init__(self):
		self.new_weight_range = 10.0
		self.weight_variation = 0.1

		self.probabilities = []

		self.probabilities.append(Probability(MutationType.NEW_CONNECTION, 0.1))
		self.probabilities.append(Probability(MutationType.NEW_NODE, 0.2))
		self.probabilities.append(Probability(MutationType.MODIFY_WEIGHT, 0.3))

	def mutate(self, network):
		genome = copy.deepcopy(Node.genome)
		rand_num = random.random()

		for probability in self.probabilities:
			if probability.value > rand_num:
				if probability.type.MutationType.NEW_CONNECTION:
					self.add_new_connection(genome)
				elif probability.type.MutationType.NEW_NODE:
					self.add_new_node(genome)
				elif probability.type.MutationType.MODIFY_WEIGHT:
					self.modify_connection_weight(genome)

		return Network(genome)

	def add_new_connection(self, genome):
		node1 = random.choice(genome.nodes)
		node2 = random.choice(genome.nodes)
		weight = self.new_weight_range - 2.0*self.new_weight_range*random.random()

		genome.add_new_connection(node1.id, node2.id, weight)

	def add_new_node(self, genome):
		gene_id = random.choice(range(len(genome.genes)))

		genome.add_new_node(gene_id)

	def change_connection_status(self, genome):
		gene = random.choice(genome.genes)

		gene.enabled = not gene.enabled

	def modify_connection_weight(self, genome):
		gene = random.choice(genome.genes)

		gene.weight *= 1.0 + self.weight_variation*(1.0 - 2.0*random.random())

class MutationType(Enum):
	NEW_CONNECTION = 0
	NEW_NODE = 1
	MODIFY_WEIGHT = 2

class Probability:
	def __init__(self, prob_type, value):
		self.type = prob_type
		self.value = value