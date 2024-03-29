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
from . import genome as gen

class Network:
	def __init__(self, genome = None):
		self.nodes = []
		self.output_node_ids = []
		self.genome = None
		self.fitness = 0

		if genome:
			self.set_up(genome)

	def __lt__(self, other):
		return self.fitness < other.fitness

	def set_up(self, genome):
		self.nodes = []
		self.output_node_ids = []
		self.genome = genome

		self.__add_nodes()
		self.__connect_nodes()

	def __add_nodes(self):
		for node in self.genome.nodes:
			if node.type == gen.NodeType.BIAS:
				self.nodes.append(BiasNode())

			elif node.type == gen.NodeType.INPUT:
				self.nodes.append(InputNode(node.id))

			elif node.type == gen.NodeType.HIDDEN:
				loc_node = HiddenNode(node.id)
				loc_node.level = node.level

				self.nodes.append(loc_node)

			elif node.type == gen.NodeType.OUTPUT:
				loc_node = OutputNode(node.id)
				loc_node.level = node.level

				self.output_node_ids.append(node.id)
				self.nodes.append(loc_node)

	def __connect_nodes(self):
		for gene in self.genome.genes:
			if gene.used:
				connection = Connection(self.nodes[gene.in_node_id], gene.weight, gene.enabled)
				self.nodes[gene.out_node_id].connections.append(connection)

	def execute(self, input_values_node_ids):
		ret_vals = {}
		self.__reset_node_values()

		for value, node_id in input_values_node_ids:
			self.nodes[node_id].value = value

		for node_id in self.output_node_ids:
			ret_vals[node_id] = self.nodes[node_id].execute()

		return ret_vals

	def __reset_node_values(self):
		for node in self.nodes:
			node.reset_value()

class Node:
	def __init__(self, node_id):
		self.id = node_id
		self.connections = []
		self.value = 0
		self.type = None
		self.level = 0

	def __str__(self):
		string = ""

		string += "ID: {:3d} ".format(self.id)
		string += "TYPE: {:6s} ".format(self.type.name)
		string += "LEVEL: {:3d} ".format(self.level)

		string += "CONNECTED NODES: ["

		for connection in self.connections:
			string += "("
			string += str(connection.node.id) + ", "
			string += str(round(connection.weight, 3)) + ", "
			string += str(connection.enabled)
			string += "), "

		string += "]"

		return string

	def reset(self):
		self.connections = []
		self.value = 0

	def reset_value(self):
		self.value = 0

	def reset_connection(self):
		self.connections = []

	def execute(self):
		for connection in self.connections:
			if connection.enabled:
				self.value += connection.weight*connection.node.execute()

		return utility.modified_sigmoid(self.value)

class OutputNode(Node):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.type = NodeType.OUTPUT
		self.level = 1

class HiddenNode(Node):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.type = NodeType.HIDDEN

class InputNode(Node):
	def __init__(self, node_id):
		super().__init__(node_id)
		self.type = NodeType.INPUT

	def execute(self):
		return self.value

class BiasNode(Node):
	def __init__(self):
		super().__init__(node_id = 0)
		self.type = NodeType.BIAS
		self.value = 1.0

	def reset(self):
		self.connections = []
		self.value = 1.0

	def reset_value(self):
		self.value = 1.0

	def execute(self):
		return 1.0

	def get_level(self):
		return 0

class NodeType(Enum):
	BIAS = 0
	INPUT = 1
	HIDDEN = 2
	OUTPUT = 3

class Connection:
	def __init__(self, node, weight = 1.0, enabled = True):
		self.node = node
		self.weight = weight
		self.enabled = enabled