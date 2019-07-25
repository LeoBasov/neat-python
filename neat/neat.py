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
from enum import Enum

from .network import Network

class NEAT:
	def __init__(self):
		pass

class Species:
	def __init__(self, genome, c1 = 1.0, c2 = 1.0, c3 = 1.0):
		self.genome = genome

		self.c1 = c1
		self.c2 = c2
		self.c3 = c3

	def compare(self, genome):
		N = max(len(self.genome.genes), len(genome.genes))
		max_innovation = self.find_max(genome)
		lists = self.set_up_lists(genome, max_innovation)
		distance = self.calc_distance(lists, N)

		return distance

	def find_max(self, genome):
		max_innovation = 0

		for gene in self.genome.genes:
			if gene.innovation > max_innovation:
				max_innovation = gene.innovation

		for gene in genome.genes:
			if gene.innovation > max_innovation:
				max_innovation = gene.innovation

		return max_innovation + 1

	def set_up_lists(self, genome, max_innovation):
		lists = max_innovation*[None]
		list_self = max_innovation*[None]
		list_other = max_innovation*[None]

		for gene in self.genome.genes:
			list_self[gene.innovation] = gene

		for gene in genome.genes:
			list_other[gene.innovation] = gene

		for i in range(max_innovation):
			lists[i] = (list_self[i], list_other[i])

		return lists

	def calc_distance(self, lists, N):
		number_of_excesses = 0
		number_of_disjoints = 0
		number_of_matching = 0
		average_weight_matching = 0.0
		excess = True

		for element in reversed(lists):
			if element[0] and element[1]:
				excess = False
				average_weight_matching += 0.5*(element[0].weight + element[1].weight)
				number_of_matching += 1

			elif (element[0] and not element[1]) or (not element[0] and element[1]) and excess:
				number_of_excesses += 1

			elif (element[0] and not element[1]) or (not element[0] and element[1]):
				number_of_disjoints += 1

		return self.c1*(number_of_excesses/N) + self.c2*(number_of_disjoints/N) + self.c3*(average_weight_matching/number_of_matching)

class Mutator:
	def __init__(self):
		self.new_weight_range = 10.0
		self.weight_variation = 0.1
		self.max_network_size = 5

		self.probabilities = []

		self.probabilities.append(Probability(MutationType.NEW_CONNECTION, 0.2))
		self.probabilities.append(Probability(MutationType.NEW_NODE, 0.01))
		self.probabilities.append(Probability(MutationType.MODIFY_WEIGHT, 0.5))
		self.probabilities.append(Probability(MutationType.CHANGE_CONNECTION_STATUS, 0.33))
		self.probabilities.append(Probability(MutationType.NEW_WEIGHT, 0.31))

		self.probabilities.sort()

	def mutate(self, network):
		genome = copy.deepcopy(network.genome)
		rand_num = random.random()

		for probability in self.probabilities:
			if probability.value > rand_num:
				if probability.type == MutationType.NEW_CONNECTION:
					self.add_new_connection(genome)
				elif probability.type == MutationType.NEW_NODE and len(genome.nodes) < self.max_network_size:
					self.add_new_node(genome)
				elif probability.type == MutationType.MODIFY_WEIGHT:
					self.modify_connection_weight(genome)
				elif probability.type == MutationType.CHANGE_CONNECTION_STATUS:
					self.change_connection_status(genome)
				elif probability.type == MutationType.NEW_WEIGHT:
					self.set_new_connection_weight(genome)

				break

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

	def set_new_connection_weight(self, genome):
		gene = random.choice(genome.genes)

		gene.weight = self.new_weight_range - 2.0*self.new_weight_range*random.random()

class MutationType(Enum):
	NEW_CONNECTION = 0
	NEW_NODE = 1
	MODIFY_WEIGHT = 2
	NEW_WEIGHT = 3
	CHANGE_CONNECTION_STATUS = 4

class Probability:
	def __init__(self, prob_type, value):
		self.type = prob_type
		self.value = value

	def __lt__(self, other):
		return self.value < other.value