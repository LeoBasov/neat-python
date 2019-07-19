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

import math
from enum import Enum

from . import utility

class Network:
	def __init__(self):
		self.nodes = {0 : BiasNode()} #Bias node has always the id = 0
		self.input_node_ids = []
		self.output_node_ids = []
		self.genes = []
		self.max_level = 0

	def set_genes(self, genes):
		self.genes = genes
		self._reset_node_connections()

		for gene in self.genes:
			self._add_hidden_node(gene.in_node, gene.out_node)
			self._connect_gene(gene)

		self._get_max_level()
		self._adjust_level()

	def _adjust_level(self):
		for node_id in self.output_node_ids:
			self.nodes[node_id]._adjust_level(self.max_level)

	def _get_max_level(self):
		self.max_level = 0

		for node_id in self.output_node_ids:
			self.max_level = max(self.nodes[node_id]._get_level(self.max_level), self.max_level)

	def _add_hidden_node(self, in_node_id, out_node_id):
		if out_node_id not in self.nodes:
			self.nodes[out_node_id] = HiddenNode(out_node_id)

	def _add_input_node(self, node_id):
		if node_id not in self.nodes and node_id not in self.input_node_ids:
			self.input_node_ids.append(node_id)
			self.nodes[node_id] = InputNode(node_id)

	def _add_output_node(self, node_id):
		if node_id not in self.nodes and node_id not in self.output_node_ids:
			self.output_node_ids.append(node_id)
			self.nodes[node_id] = OutputNode(node_id)

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
		self.level = 0

	def __str__(self):
		input_nodes_weights = []

		for node_weight in self.in_nodes_weights:
			input_nodes_weights.append((node_weight[0].id, node_weight[1]))

		return "ID: " + str(self.id) + ". Level: " + str(self.level) + ". Input node-weights: " + str(input_nodes_weights) + ". Value: " + str(self.value) + "."

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

		return utility.sigmoid(self.value)

	def _adjust_level(self, level):
		self.level = level

		for node, weight in self.in_nodes_weights:
			node._adjust_level(level - 1)

	def _get_level(self, level):
		loc_level = 0

		for node, weight in self.in_nodes_weights:
			loc_level = max(node._get_level(level), loc_level)

		return level + loc_level + 1

class OutputNode(Node):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.type = NodeType.OUTPUT_NODE
		self.level = None

	def _adjust_level(self, level):
		for node, weight in self.in_nodes_weights:
			node._adjust_level(level)

	def _get_level(self, level):
		loc_level = 0

		for node, weight in self.in_nodes_weights:
			loc_level = max(node._get_level(level), loc_level)

		return level + loc_level

class HiddenNode(Node):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.type = NodeType.HIDDEN_NODE

class InputNode(Node):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.type = NodeType.INPUT_NODE

	def execute(self):
		return self.value

	def _adjust_level(self, level):
		pass

	def _get_level(self, level):
		return level

class BiasNode(Node):
	def __init__(self):
		super().__init__(node_id = 0)
		self.type = NodeType.BIAS_NODE
		self.value = 1.0

	def reset(self):
		self.in_nodes_weights = []
		self.value = 1.0

	def reset_value(self):
		self.value = 1.0

	def execute(self):
		return 1.0

	def _adjust_level(self, level):
		pass

	def _get_level(self, level):
		return level

class NodeType(Enum):
	BIAS_NODE = 0
	INPUT_NODE = 1
	HIDDEN_NODE = 2
	OUTPUT_NODE  = 3

class Gene:
	def __init__(self, in_node = None, out_node = None, weight = 1.0, enabled = False):
		self.in_node = in_node
		self.out_node = out_node
		self.weight = weight
		self.enabled = enabled