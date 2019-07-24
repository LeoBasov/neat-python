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
		self.nodes = []
		self.output_node_ids = []

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

		string += "ID: " + str(self.id) + " "
		string += "TYPE: " + str(self.type.name) + " "
		string += "LEVEL: " + str(self.level) + " "

		string += "CONNECTED NODES: ["

		for connection in self.connections:
			string += str(connection.node.id) + ", "

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

		return utility.sigmoid(self.value)

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