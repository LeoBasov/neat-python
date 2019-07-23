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

from enum import Enum

class NodeType(Enum):
	BIAS = 0
	INPUT = 1
	HIDDEN = 2
	OUTPUT = 3

class Genome:
	def __init__(self):
		self.nodes = [BiasNode()]
		self.output_nodes_ids = []
		self.genes = []

	def add_input_node(self):
		node_id = len(self.nodes) + 1
		node = InputNode(node_id)

		self.nodes.append(node)

		return node_id

	def add_output_node(self):
		node_id = len(self.nodes) + 1
		node = OutputNode(node_id)

		self.output_nodes_ids.append(node_id)
		self.nodes.append(node)

		return node_id

	def add_hidden_node(self):
		node_id = len(self.nodes) + 1
		node = HiddenNode(node_id)

		self.nodes.append(node)

		return node_id

class Node:
	def __init__(self, node_id, node_type):
		self.id = None
		self.type = None
		self.level = 0
		self.connected_nodes = []

	def update_level(self):
		loc_level = 0

		for node in self.connected_nodes:
			loc_level = max(node.update_level(), loc_level)

		self.level = loc_level + 1

		return self.level

class BiasNode(Node):
	def __init__(self):
		super().__init__(0, NodeType.BIAS)

	def update_level(self):
		return 0

class HiddenNode(Node):
	def __init__(self, node_id, node_type = NodeType.HIDDEN):
		super().__init__(node_id, node_type)

		if node_id == 0:
			raise Error("HiddenNode.___init__", "Node id can not be 0. Reserved for bias node")
		elif node_id < 0:
			raise Error("HiddenNode.___init__", "Node id can not < 0.")

class InputNode(HiddenNode):
	def __init__(self, node_id):
		super().__init__(node_id, NodeType.INPUT)

	def update_level(self):
		return 0

class OutputNode(HiddenNode):
	def __init__(self, node_id):
		super().__init__(node_id, NodeType.OUTPUT)

class Gene:
	def __init__(self, in_node_id, out_node_id, weight = 1.0, enabled = True):
		self.in_node_id = in_node_id
		self.out_node_id = out_node_id
		self.weight = weight
		self.enabled = enabled

	def connection_exists(self, in_node_id, out_node_id):
		forward_connection = (self.in_node_id == in_node_id) and (self.out_node_id == out_node_id)
		backward_connection = (self.in_node_id == out_node_id) and (self.out_node_id == in_node_id)

		return forward_connection or backward_connection

class Error(Exception):
	"""Base Exception for this module.

	Attributes
	----------
	expression : str
		input expression in which the error occurred
	message : str
		explanation of the error

	"""

	def __init__(self, expression, message):
		self.expression = "genome." + expression
		self.message = message