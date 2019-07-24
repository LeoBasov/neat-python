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
		self.weight_variance = 10.0

	def mutate(self, network):
		genome = copy.deepcopy(Node.genome)

		#mutate genome

		return Network(genome)

	def add_new_connection(self, genome):
		node1 = random.choice(genome.nodes)
		node2 = random.choice(genome.nodes)
		weight = self.weight_variance - 2.0*self.weight_variance*random.random()

		genome.add_new_connection(node1.id, node2.id, weight)

	def add_new_node(self, genome):
		gene_id = random.choice(range(len(genome.genes)))

		genome.add_new_node(gene_id)

	def change_connection_status(self, genome):
		gene = random.choice(genome.genes)

		gene.enabled = not gene.enabled