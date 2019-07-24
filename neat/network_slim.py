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

class Node:
	def __init__(self, node_id):
		self.id = node_id
		self.connections = []
		self.value = 0

	def __str__(self):
		input_nodes_weights = []

		for connection in self.connections:
			input_nodes_weights.append((connection.node.id, connection.weight, connection.enabled))

		return "ID: " + str(self.id) + ". Level: " + str(self.level) + ". Input node-weights: " + str(input_nodes_weights) + ". Value: " + str(self.value) + "."

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
		self.level = None

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
		self.type = NodeType.BIAS_NODE
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
	def __init__(self):
		self.node = None
		self.weight = 1.0
		self.enabled = True