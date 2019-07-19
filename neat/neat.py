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
		self.new_node_prob = 1.0
		self.new_connection_prob = 0
		self.new_activation_status_prob = 0.0

		self.weight_modification_variation = 0.1
		self.weight_setting_variation = 10.0

	def mutate(self, network):
		new_network = copy.deepcopy(network)
		"""rand_nr = random.random()
		genes = new_network.genes
		nodes = new_network.nodes
		node_id = None

		self._change_weights(genes)

		if rand_nr < self.new_node_prob:
			(node_id, genes) = self._generate_new_node(genes, nodes)
		elif rand_nr < self.new_connection_prob:
			genes = self._generate_new_connection(genes, nodes)
		elif rand_nr < self.new_activation_status_prob:
			genes = self._generate_new_connection_status(genes)

		if node_id:
			new_network._add_hidden_node(node_id)

		new_network.set_genes(genes)"""

		return new_network

	def _modify_weight(self, network):
		genes = network.genes
		gene = random.choice(genes)
		gene.weight *= 1.0 + self.weight_modification_variation*(1.0 - 2.0*random.random())

		network.set_genes(genes)

	def _set_new_random_weight(self, network):
		genes = network.genes
		gene = random.choice(genes)
		gene.weight = 1.0 + self.weight_setting_variation*(1.0 - 2.0*random.random())

		network.set_genes(genes)

	def _set_new_random_weight_all(self, network):
		genes = network.genes
		
		for gene in genes:
			gene.weight = 1.0 + self.weight_setting_variation*(1.0 - 2.0*random.random())

		network.set_genes(genes)

	def _generate_new_node(self, network):
		new_node_id = self._get_net_network_node_id(network)
		genes = network.genes
		gene = random.choice(genes)
		gene.enabled = False
		gene1 = Gene(in_node = gene.in_node, out_node = new_node_id, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = new_node_id, out_node = gene.out_node, weight = 1.0, enabled = True)		

		genes.append(gene1)
		genes.append(gene2)

		network.set_genes(genes)

	def _get_net_network_node_id(self, network):
		new_node_id = 0

		for key, node in network.nodes.items():
			if node.id > new_node_id:
				new_node_id = node.id

		return new_node_id + 1

	def _generate_new_connection(self, genes, nodes):
		"""in_node = random.choice(nodes)
		out_node = random.choice(nodes)

		while (in_node == out_node) or (in_node.type == NodeType.OUTPUT_NODE) or (out_node.type == NodeType.INPUT_NODE) or (out_node.type == NodeType.BIAS_NODE):
			in_node = random.choice(nodes)
			out_node = random.choice(nodes)


		for gene in genes:
			if (in_node.id == gene.in_node and out_node.id == gene.out_node) or (in_node.id == gene.out_node and out_node.id == gene.out_node):
				return genes

		gene = Gene(in_node = in_node.id, out_node = out_node.id, weight = 1.0, enabled = True)

		genes.append(gene)"""

		return genes

	def _midifiy_connection_status(self, network):
		genes = network.genes
		gene = random.choice(genes)
		gene.enabled = not gene.enabled

		network.set_genes(genes)