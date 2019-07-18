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
		self.new_node_prob = 0.3
		self.new_connection_prob = 0.4
		self.new_activation_status_prob = 0.5

		self.weight_variation = 0.1

	def mutate(self, network):
		new_network = copy.deepcopy(network)
		rand_nr = random.random()
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

		new_network.set_genes(genes)

		return new_network

	def _change_weights(self, genes):
		gene = random.choice(genes)
		gene.weight = gene.weight*(1.0 + self.weight_variation - 2.0*self.weight_variation*random.random())

		return genes

	def _generate_new_node(self, genes, nodes):
		new_node_id = max(list(nodes.keys())) + 1
		gene = random.choice(genes)

		gene1 = Gene(in_node = gene.in_node, out_node = new_node_id, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = new_node_id, out_node = gene.out_node, weight = 1.0, enabled = True)

		gene.enabled = False

		genes.append(gene1)
		genes.append(gene2)

		return (new_node_id, genes)

	def _generate_new_connection(self, genes, nodes):
		in_node = random.choice(nodes)
		out_node = random.choice(nodes)

		while in_node == out_node:
			in_node = random.choice(nodes)
			out_node = random.choice(nodes)

		if in_node.type == NodeType.OUTPUT_NODE:
			return genes
		elif out_node.type == NodeType.INPUT_NODE:
			return genes

		for gene in genes:
			if in_node.id == gene.in_node or out_node.id == gene.out_node or in_node.id == gene.out_node or out_node.id == gene.out_node:
				return genes

		gene = Gene(in_node = in_node.id, out_node = out_node.id, weight = 1.0, enabled = True)

		genes.append(gene)

		return genes

	def _generate_new_connection_status(self, genes):
		gene = random.choice(genes)
		gene.enabled = not gene.enabled

		return genes