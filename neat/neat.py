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
import time
import os
import csv

from .network import Network
from .genome import Genome

class NEAT:
	def __init__(self):
		self.file_dir = './output/' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
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
	def initialze_network(self, **kwargs):
		pass

	def evaluate_network(self, network):
		return 0

	def evaluate_best_network(self, network):
		return [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

	#--------------------------------------------------------------

	def initiatlize(self, **kwargs):
		self.mutator.new_weight_range = kwargs['new_weight_range']

		self.number_itterations = kwargs["number_itterations"]
		self.number_sub_cycles = kwargs["number_sub_cycles"]

		self.test_case_name = kwargs["test_case_name"]
		self.test_case_specifics = kwargs["test_case_specifics"]

		self.initialze_networks(**kwargs)

	def initialze_networks(self, **kwargs):
		number_networks = kwargs['number_networks']

		for _ in range(number_networks):
			self.networks.append(self.initialze_network(**kwargs))

	def evaluate_networks(self):
		for network in self.networks:
			mean_fitness = 0.0

			for _ in range(self.number_sub_cycles):
				mean_fitness += self.evaluate_network(network)

			network.fitness = mean_fitness/(self.number_sub_cycles)

		self.networks.sort()
		self.networks.reverse()

	def evaluate_species(self):
		for species in self.species:
			species.counter += 1

			for network_id in species.networks:
				if self.networks[network_id].fitness > species.max_fitness:
					species.max_fitness = self.networks[network_id].fitness
					species.counter = 0

		rest_species = []

		for species in self.species:
			if species.counter < species.unimproved_life_time:
				rest_species.append(species)

		self.species = rest_species

	def mutate(self):
		copy_list = [elem for elem in self.species]

		for species in self.species:
			if len(species.networks) < 2:
				continue

			rest = int(round(0.5*len(species.networks)))

			for i in range(rest, len(species.networks)):
				rand_num =  random.random()

				if len(self.species) > 2 and rand_num < 0.1 and species.genome != 0:
					random.shuffle(copy_list)
					found = False

					for parent_species in copy_list:
						if len(parent_species.networks) > 2 and parent_species.genome != species.genome and parent_species.genome != 0:
							found = True
							break

					if not found:
						continue

					parent_id_1 = random.choice(parent_species.networks[:rest])
					parent_id_2 = random.choice(species.networks[:rest])

					self.networks[species.networks[i]] = copy.deepcopy(self.networks[parent_id_2])

					genome_parent_1 = self.networks[parent_id_1].genome
					genome_parent_2 = self.networks[parent_id_2].genome
					genome_child = self.networks[species.networks[i]].genome

					Genome.mate(genome_parent_1, genome_parent_2, genome_child)
					self.networks[species.networks[i]].set_up(genome_child)

				else:
					net_id  = random.choice(species.networks[:rest])
					self.networks[species.networks[i]] = copy.deepcopy(self.networks[net_id])
					self.mutator.mutate(self.networks[species.networks[i]])

	def start(self, **kwargs):
		Genome.reset()
		Species.reset()

		self.initiatlize(**kwargs)

		Genome.MAX_NUMBER_GENES = len(self.networks[0].genome.genes)

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
			print("EXPECTED VALUE: {} CALCULATED VALUE: {} FITNESS: {}".format(vals[0], vals[1], vals[2]))

		print(80*"-")

		for node in best_network.nodes:
			print(node)

		print(80*"-")

		for gene in best_network.genome.genes:
			print(gene)

		print(80*"-")

	def print_footer(self):
		print(80*"-")
		print("Execution finished")
		print(80*"=")

	def main_loop(self):
		for i in range(self.number_itterations):
			mean_fitness = 0

			self.mutate()
			self.evaluate_networks()
			#self.evaluate_species()
			self.sort_in_species()
			self.write_to_file(i)

			for network in self.networks:
				mean_fitness += network.fitness

			mean_fitness /= len(self.networks)

			print("MEAN FITNESS: %0.3f SPECIES NUMBER : %3.0d PERFORMED ITTERATIONS %0.0d/%0.0d" % (round(mean_fitness,3), len(self.species), i + 1, self.number_itterations), end="\r", flush=True)

		print("")
		print(80*"-")

	def sort_in_species(self):
		for species in self.species:
				species.networks.clear()

		for i in range(len(self.networks)):
			found = False

			for species in self.species:
				if species.compare(self.networks[i].genome):
					species.networks.append(i)
					found = True
					break

			if not found:
				species  = Species(self.networks[i].genome)
				species.networks.append(i)
				self.species.append(species)

	def write_to_file(self, step):
		try:
			os.makedirs(self.file_dir)
			os.makedirs(self.file_dir + '/species')
			os.makedirs(self.file_dir + '/networks')

		except FileExistsError:
			pass

		finally:
			self.write_fitness(self.file_dir, step)
			self.write_species(self.file_dir + '/species', step)
			self.write_networks(self.file_dir + '/networks', step)

	def write_networks(self, file_dir, step):
		for i in range(len(self.networks)):
			file_name = file_dir + '/network_' + str(i) + '.csv'
			row = []

			for species in self.species:
				if i in species.networks:
					row.append(int(species.id))

			for gene in self.networks[i].genome.genes:
				row.append(int(gene.used))

				if gene.used:
					row.append(int(gene.innovation))
					row.append(int(gene.enabled))
					row.append(int(gene.in_node_id))
					row.append(int(gene.out_node_id))
					row.append(float(gene.weight))
				else:
					row.append(int(gene.innovation))
					row.append(int(gene.enabled))
					row.append(0)
					row.append(0)
					row.append(float(gene.weight))

			with open(file_name, 'a') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(row)

	def write_species(self, file_dir, step):
		for species in self.species:
			file_name = file_dir + '/species_' + str(species.genome) + '.csv'
			row = [step, len(species.networks)]

			with open(file_name, 'a') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(row)

	def write_fitness(self, file_dir, step):
		file_name = file_dir + '/fitness.csv'
		row = []

		for network in self.networks:
			row.append(network.fitness)

		with open(file_name, 'a') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(row)

	def write_header(self, file_name, field_names):
		if not os.path.isfile(file_name):
			with open(file_name, 'a') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(field_names)

class Species:
	ID = 0

	def reset():
		Species.ID = 0

	def __init__(self, genome, c1 = 1.0, c2 = 1.0, c3 = 0.4, unimproved_life_time = 15):
		Species.ID += 1

		self.id = Species.ID
		self.genome = Genome.get_species(genome)
		self.unimproved_life_time = unimproved_life_time
		self.counter = 0
		self.max_fitness = 0.0

		self.c1 = c1
		self.c2 = c2
		self.c3 = c3

		self.networks = []

	def compare(self, genome):
		return self.genome == Genome.get_species(genome)

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
		self.probabilities.append(Probability(MutationType.CHANGE_CONNECTION_STATUS, 0.01))
		self.probabilities.append(Probability(MutationType.NEW_WEIGHT, 0.08))

		self.probabilities.sort()

	def mutate(self, network):
		genome = network.genome
		rand_num = random.random()

		for probability in self.probabilities:
			if probability.value > rand_num:
				if probability.type == MutationType.NEW_CONNECTION and genome.unused_gene_index != len(genome.genes):
					self.add_new_connection(genome)
					network.set_up(genome)
					break

				elif probability.type == MutationType.NEW_NODE and (len(genome.unused_nodes_ids) != genome.unused_nodes_current_id):
					self.add_new_node(genome)
					network.set_up(genome)
					break

				elif probability.type == MutationType.MODIFY_WEIGHT:
					self.modify_connection_weight(genome)
					network.set_up(genome)
					break

				elif probability.type == MutationType.CHANGE_CONNECTION_STATUS:
					self.change_connection_status(genome)
					network.set_up(genome)
					break

				elif probability.type == MutationType.NEW_WEIGHT:
					self.set_new_connection_weight(genome)
					network.set_up(genome)
					break

	def add_new_connection(self, genome):
		node1 = random.choice(genome.nodes)
		node2 = random.choice(genome.nodes)
		weight = self.new_weight_range - 2.0*self.new_weight_range*random.random()

		while node1.id in genome.unused_nodes_ids:
			node1 = random.choice(genome.nodes)

		while node2.id in genome.unused_nodes_ids:
			node2 = random.choice(genome.nodes)

		genome.add_new_connection(node1.id, node2.id, weight)

	def add_new_node(self, genome):
		gene_id = random.choice(range(len(genome.genes)))

		while not genome.genes[gene_id].used:
			gene_id = random.choice(range(len(genome.genes)))

		genome.add_new_node(gene_id)

	def change_connection_status(self, genome):
		gene = random.choice(genome.genes)

		while not gene.used:
			gene = random.choice(genome.genes)

		gene.enabled = not gene.enabled

	def modify_connection_weight(self, genome):
		gene = random.choice(genome.genes)

		while not gene.used:
			gene = random.choice(genome.genes)

		gene.weight *= 1.0 + self.weight_variation*(1.0 - 2.0*random.random())

	def set_new_connection_weight(self, genome):
		gene = random.choice(genome.genes)

		while not gene.used:
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
