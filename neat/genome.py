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

class Genome:
	pass

class Node:
	def __init__(self, node_id, node_type):
		self.id = None
		self.type = None

class BiasNode(Node):
	def __init__(self):
		super().__init__(node_id = 0, node_type = NodeType.BIAS)

class OtherNode(Node):
	def __init__(self, node_id, node_type):
		super().__init__(node_id = node_id, node_type = node_type)

		if node_id == 0:
			raise Error("OtherNode.___init__", "Node id can not be 0. Reserved for bias node")
		elif node_id < 0:
			raise Error("OtherNode.___init__", "Node id can not < 0.")

class Gene:
	pass

class NodeType(Enum):
	BIAS = 0
	INPUT = 1
	HIDDEN = 2
	OUTPUT = 3

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