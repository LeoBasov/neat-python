"""live_sim a evolution simulation
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

from .network import Node
from .network import Gene
from .network import Network
from .network import NodeType

class NEAT:
	def __init__(self):
		self.new_node_prob = 0.01
		self.new_connection_prob = 0.2
		self.set_new_weight_prob = 0.31
		self.new_activation_status_prob = 0.33
		self.modify_weight_prob = 0.5
		self.max_network_size = 7

		self.weight_modification_variation = 0.1
		self.weight_setting_variation = 10.0

	def mutate(self, network):
		new_network = copy.deepcopy(network)
		rand_nr = random.random()

		if rand_nr < self.new_node_prob and len(new_network.nodes) < self.max_network_size :
			self.__generate_new_node(new_network)

		elif rand_nr < self.new_connection_prob:
			self.__generate_new_connection(new_network)

		elif rand_nr < self.set_new_weight_prob:
			self.__set_new_random_weight(new_network)

		elif rand_nr < self.new_activation_status_prob:
			self.__midifiy_connection_status(new_network)

		elif rand_nr < self.modify_weight_prob:
			self.__modify_weight(new_network)

		return new_network

	def __modify_weight(self, network):
		genes = network.genes
		gene = random.choice(genes)
		gene.weight *= 1.0 + self.weight_modification_variation*(1.0 - 2.0*random.random())

		network.set_genes(genes)

	def __set_new_random_weight(self, network):
		genes = network.genes
		gene = random.choice(genes)
		gene.weight = 1.0 + self.weight_setting_variation*(1.0 - 2.0*random.random())

		network.set_genes(genes)

	def __set_new_random_weight_all(self, network):
		genes = network.genes
		
		for gene in genes:
			gene.weight = 1.0 + self.weight_setting_variation*(1.0 - 2.0*random.random())

		network.set_genes(genes)

	def __generate_new_node(self, network):
		new_node_id = self.__get_net_network_node_id(network)
		genes = network.genes
		gene = random.choice(genes)
		gene.enabled = False
		gene1 = Gene(in_node = gene.in_node, out_node = new_node_id, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = new_node_id, out_node = gene.out_node, weight = 1.0, enabled = True)		

		genes.append(gene1)
		genes.append(gene2)

		network.set_genes(genes)

	def __get_net_network_node_id(self, network):
		new_node_id = 0

		for key, node in network.nodes.items():
			if node.id > new_node_id:
				new_node_id = node.id

		return new_node_id + 1

	def __generate_new_connection(self, network):
		node_ids = list(network.nodes.keys())
		random.shuffle(node_ids)
		genes = network.genes
		found_possible_connection = False
		node_in = None
		node_out = None

		for node_in_id in node_ids:
			for node_out_id in node_ids:
				node_in = network.nodes[node_in_id]
				node_out = network.nodes[node_out_id]

				if self.__check_connection_creteia(node_in, node_out, genes):
					found_possible_connection = True
					break

			else:
				continue
			break

		if found_possible_connection:
			gene = Gene(in_node = node_in.id, out_node = node_out.id, weight = 1.0, enabled = True)
			genes.append(gene)
			network.set_genes(genes)

	def __check_connection_creteia(self, node_in, node_out, genes):
		#Input output node criteria
		if (node_in.type == NodeType.OUTPUT_NODE) or (node_out.type == NodeType.INPUT_NODE) or (node_out.type == NodeType.BIAS_NODE):
			return False

		#existence criteria
		for gene in genes:
			if (node_in.id == gene.in_node and node_out.id == gene.out_node) or (node_in.id == gene.out_node and node_out.id == gene.in_node):
				return False

		#no circular dependecy criteria
		if (node_out.type != NodeType.OUTPUT_NODE) and (node_in.level >= node_out.level):
			return False

		return True

	def __midifiy_connection_status(self, network):
		genes = network.genes
		gene = random.choice(genes)
		gene.enabled = not gene.enabled

		network.set_genes(genes)