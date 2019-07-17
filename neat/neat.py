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

class NEAT:
	def __init__(self):
		self.new_node_prob = 0.1
		self.new_connection_prob = 0.2
		self.new_activation_status_prob = 0.3

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
		for gene in genes:
			if gene.weight == 0.0:
				gene.weight = ((1.0 + self.weight_variation) - 2.0*self.weight_variation*random.random())
			else:
				gene.weight = ((1.0 + self.weight_variation) - 2.0*self.weight_variation*random.random())*gene.weight

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

		gene = Gene(in_node = in_node.id, out_node = out_node.id, weight = 1.0, enabled = True)

		genes.append(gene)

		return genes

	def _generate_new_connection_status(self, genes):
		gene = random.choice(genes)
		gene.enabled = not gene.enabled

		return genes

class Network:
	def __init__(self):
		self.nodes = {0 : BiasNode()} #Bias node has always the id = 0
		self.input_node_ids = []
		self.output_node_ids = []
		self.genes = []

	def set_genes(self, genes):
		self.genes = genes
		self._reset_node_connections()

		for gene in self.genes:
			self._add_hidden_node(gene.in_node)
			self._add_hidden_node(gene.out_node)
			self._connect_gene(gene)

	def _add_hidden_node(self, node_id):
		if node_id not in self.nodes:
			self.nodes[node_id] = Node(node_id)

	def _add_input_node(self, node_id):
		if node_id not in self.nodes and node_id not in self.input_node_ids:
			self.input_node_ids.append(node_id)
			self.nodes[node_id] = Node(node_id)

	def _add_output_node(self, node_id):
		if node_id not in self.nodes and node_id not in self.output_node_ids:
			self.output_node_ids.append(node_id)
			self.nodes[node_id] = Node(node_id)

	def _connect_gene(self, gene):
		if gene.enabled and gene.out_node != 0:
			in_node = self.nodes[gene.in_node]
			self.nodes[gene.out_node].in_nodes_weights.append((in_node, gene.weight))

	def _reset_node_values(self):
		for node_id, node in self.nodes.items():
			node.reset_value()

	def _reset_node_connections(self):
		for node_id, node in self.nodes.items():
			node.reset_connection()


	def execute(self, input_values_node_ids):
		self._reset_node_values()

		for value, node_id in input_values_node_ids:
			self.nodes[node_id].value = value

		for node_id in self.output_node_ids:
			self.nodes[node_id].execute()

class Node:
	def __init__(self, node_id):
		self.id = node_id
		self.in_nodes_weights = []
		self.value = 0

	def __str__(self):
		input_nodes_weights = []

		for node_weight in self.in_nodes_weights:
			input_nodes_weights.append((node_weight[0].id, node_weight[1]))

		return "ID: " + str(self.id) + ". Input node-weights: " + str(input_nodes_weights) + ". Value: " + str(self.value) + "."

	def reset(self):
		self.in_nodes_weights = []
		self.value = 0

	def reset_value(self):
		self.value = 0

	def reset_connection(self):
		self.in_nodes_weights = []

	def execute(self):
		for node, weight in self.in_nodes_weights:
			self.value += weight*node.execute()

		return self.value

class BiasNode(Node):
	def __init__(self):
		super().__init__(node_id = 0)
		self.value = 1.0

	def reset(self):
		self.in_nodes_weights = []
		self.value = 1.0

	def reset_value(self):
		self.value = 1.0

class Gene:
	def __init__(self, in_node = None, out_node = None, weight = 1.0, enabled = False):
		self.in_node = in_node
		self.out_node = out_node
		self.weight = weight
		self.enabled = enabled