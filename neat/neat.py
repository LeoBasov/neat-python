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
import sys

from .network import Network
from .genome import Genome

class NEAT:
	def __init__(self):
		self.number_itterations = 1
		self.number_sub_cycles = 1
		self.species_distance_max = 3.0
		self.networks = []
		self.species = []
		self.mutator = Mutator()

		self.test_case_name = "PLACE HOLDER"
		self.test_case_specifics = ["PLACE HOLDER"]

	#--------------------------------------------------------------
	#To be implemented
	def initiatlize(self, **kwargs):
		pass

	def evaluate_network(self, network):
		return 0

	def evaluate_best_network(self, network):
		return [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

	#--------------------------------------------------------------

	def evaluate_networks(self):
		for network in self.networks:
			mean_fitness = 0.0

			for _ in range(self.number_sub_cycles):
				mean_fitness += self.evaluate_network(network)

			network.fitness = mean_fitness/self.number_sub_cycles

		self.networks.sort()
		self.networks.reverse()

	def evaluate_species(self):
		for species in self.species:
			species.counter += 1

			for network in species.networks:
				if network.fitness > species.max_fitness:
					species.max_fitness = network.fitness
					species.counter = 0

	def mutate(self):
		if len(self.species):
			self.networks.clear()

		rest_species = []

		for species in self.species:
			if species.counter < species.unimproved_life_time:
				rest_species.append(species)
				rest = int(max(0.5*len(species.networks), 1))
				rest_network = species.networks[:rest]

				for i in range(len(species.networks)):
					species.networks[i] = self.mutator.mutate(random.choice(rest_network))

			self.networks += species.networks

		self.species = rest_species


	def start(self, **kwargs):
		self.initiatlize(**kwargs)

		self.print_header()
		self.print_set_up()

		self.main_loop()

		self.print_best()
		self.print_footer()

	def print_header(self):
		print(80*"=")
		print("NEAT - NeuroEvolution of Augmenting Topologies")
		print("(c) 2019, Leo Basov")
		print(80*"-")
		print(self.test_case_name)

		for specific in self.test_case_specifics:
			print(specific)

		print(80*"-")

	def print_set_up(self):
		print("NUMBER_NETWORKS", len(self.networks))
		print("NUMBER_ITTERATIONS", self.number_itterations)
		print("NUMBER_SUB_CYCLES", self.number_sub_cycles)
		print(80*"-")

	def print_best(self):
		best_fintess = self.networks[0].fitness if len(self.networks) > 0 else 0
		best_network = self.networks[0] if len(self.networks) > 0 else 0

		print("FITNESS OF BEST NETWORK:", best_fintess)
		print(80*"-")

		fitness_real_calc = self.evaluate_best_network(best_network)

		for vals in fitness_real_calc:
			print("EXPECTED VALUE: {} CALCULATED VALUE: {} FITNESS: {}".format(int(vals[0]), int(round(vals[1], 0)), round(vals[2], 3)))

		print(80*"-")

		for node in best_network.nodes:
			print(node)

		print(80*"-")

		for gene in best_network.genome.genes:
			print(gene)

	def print_footer(self):
		print(80*"-")
		print("Execution finished")
		print(80*"=")

	def main_loop(self):
		for i in range(self.number_itterations):
			mean_fitness = 0
			Genome.LAST_GENES = []

			self.evaluate_networks()
			self.evaluate_species()

			for network in self.networks:
				mean_fitness += network.fitness

			mean_fitness /= len(self.networks)

			print("MEAN FITNESS: %0.3f SPECIES NUMBER : %3.0d PERFORMED ITTERATIONS %0.0d/%0.0d" % (round(mean_fitness,3), len(self.species), i + 1, self.number_itterations), end="\r", flush=True)
			
			self.mutate()
			self.sort_in_species()

		self.evaluate_networks()
		print("")
		print(80*"-")

	def sort_in_species(self):
		for species in self.species:
				species.networks.clear()
				
		for network in self.networks:
			min_distance_species = [sys.float_info.max, None]

			for species in self.species:
				distance = species.compare(network.genome)

				if min_distance_species[0] > distance:
					min_distance_species[0] = distance
					min_distance_species[1] = species

			if min_distance_species[0] > self.species_distance_max:
				species  = Species(network.genome)
				species.networks.append(network)

				self.species.append(species)
			else:
				min_distance_species[1].networks.append(network)


class Species:
	def __init__(self, genome, c1 = 1.0, c2 = 1.0, c3 = 0.4, unimproved_life_time = 15):
		self.genome = genome
		self.unimproved_life_time = unimproved_life_time
		self.counter = 0
		self.max_fitness = 0.0

		self.c1 = c1
		self.c2 = c2
		self.c3 = c3

		self.networks = []
		
	def compare(self, genome):
		N = max(len(self.genome.genes), len(genome.genes))
		lists = Genome.set_up_lists(self.genome, genome)
		distance = self.calc_distance(lists, N)

		return distance

	def calc_distance(self, lists, N):
		number_of_excesses = 0
		number_of_disjoints = 0
		number_of_matching = 0
		average_weight_differnece = 0.0
		excess = True

		for element in reversed(lists):
			if element[0] and element[1]:
				excess = False
				average_weight_differnece += abs(element[0].weight - element[1].weight)
				number_of_matching += 1

			elif (element[0] and not element[1]) or (not element[0] and element[1]) and excess:
				number_of_excesses += 1

			elif (element[0] and not element[1]) or (not element[0] and element[1]):
				number_of_disjoints += 1

		return self.c1*(number_of_excesses/N) + self.c2*(number_of_disjoints/N) + self.c3*(average_weight_differnece/number_of_matching)

class Mutator:
	def __init__(self):
		self.new_weight_range = 10.0
		self.weight_variation = 0.1

		self.probabilities = []

		self.probabilities.append(Probability(MutationType.NEW_CONNECTION, 0.05))
		self.probabilities.append(Probability(MutationType.NEW_NODE, 0.03))
		self.probabilities.append(Probability(MutationType.MODIFY_WEIGHT, 0.72))
		self.probabilities.append(Probability(MutationType.CHANGE_CONNECTION_STATUS, 0.0))
		self.probabilities.append(Probability(MutationType.NEW_WEIGHT, 0.08))

		self.probabilities.sort()

	def mutate(self, network):
		genome = copy.deepcopy(network.genome)
		rand_num = random.random()

		for probability in self.probabilities:
			if probability.value > rand_num:
				if probability.type == MutationType.NEW_CONNECTION:
					self.add_new_connection(genome)
				elif probability.type == MutationType.NEW_NODE and (len(genome.unused_nodes_ids) != genome.unused_nodes_current_id):
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