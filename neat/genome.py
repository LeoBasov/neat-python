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
	INNOVATION = 0
	LAST_GENES = []

	def reset():
		Genome.INNOVATION = 0
		Genome.LAST_GENES = []

	def __init__(self):
		self.nodes = [BiasNode()]
		self.output_nodes_ids = []
		self.genes = []

	def add_input_node(self):
		node_id = len(self.nodes)
		node = InputNode(node_id)

		self.nodes.append(node)

		return node_id

	def add_output_node(self):
		node_id = len(self.nodes)
		node = OutputNode(node_id)

		self.output_nodes_ids.append(node_id)
		self.nodes.append(node)

		return node_id

	def add_hidden_node(self):
		node_id = len(self.nodes)
		node = HiddenNode(node_id)

		self.nodes.append(node)

		return node_id

	def set_genes(self, genes):
		self.genes = []

		for gene in genes:
			if gene not in self.genes:
				self.genes.append(gene)
				self.nodes[gene.out_node_id].connected_nodes.append(self.nodes[gene.in_node_id])
			else:
				raise Error("Genome.set_genes", "Gene defined twice")

		self.update_levels()

	def add_new_connection(self, in_node_id, out_node_id, weight = 1.0, enabled = True):
		if self.connection_possible(in_node_id, out_node_id):
			gene = Gene(in_node_id, out_node_id, weight, enabled)
			self.add_gene(gene)

			return True
		else:
			return False

	def add_new_connection_no_check(self, in_node_id, out_node_id, weight = 1.0, enabled = True):
		gene = Gene(in_node_id, out_node_id, weight, enabled)
		self.add_gene(gene)

	def connection_possible(self, in_node_id, out_node_id):
		no_in_output = self.nodes[in_node_id].type != NodeType.OUTPUT
		no_out_input = self.nodes[out_node_id].type != NodeType.INPUT
		no_level_conflict = self.nodes[in_node_id].level < self.nodes[out_node_id].level
		no_existence = True

		for gene in self.genes:
			if gene.connection_exists(in_node_id, out_node_id):
				no_existence = False
				break

		return no_in_output and no_out_input and no_level_conflict and no_existence

	def add_gene(self, gene):
		if gene not in self.genes:
			self.check_gene(gene)
			self.genes.append(gene)
			self.nodes[gene.out_node_id].connected_nodes.append(self.nodes[gene.in_node_id])

	def add_new_node(self, gene_id):
		if self.genes[gene_id].enabled:
			self.genes[gene_id].enabled = False
			in_node_id = self.genes[gene_id].in_node_id
			out_node_id = self.genes[gene_id].out_node_id
			new_node_id = self.add_hidden_node()
			weight = self.genes[gene_id].weight

			gene1 = Gene(in_node_id, new_node_id)
			gene2 = Gene(new_node_id, out_node_id, weight)

			self.add_gene(gene1)
			self.add_gene(gene2)

			self.update_levels()

	def update_levels(self):
		for node_id in self.output_nodes_ids:
			self.nodes[node_id].update_level()

	def check_gene(self, gene):
		for old_gene in Genome.LAST_GENES:
			if old_gene.connection_exists(gene.in_node_id, gene.out_node_id):
				return

		Genome.INNOVATION += 1
		Genome.LAST_GENES.append(gene)

		gene.innovation = Genome.INNOVATION

class Node:
	def __init__(self, node_id, node_type):
		self.id = node_id
		self.type = node_type
		self.level = 0
		self.connected_nodes = []

	def __str__(self):
		string = ""

		string += "ID: " + str(self.id) + " "
		string += "TYPE: " + str(self.type.name) + " "
		string += "LEVEL: " + str(self.level) + " "

		string += "CONNECTED NODES: ["

		for connected_node in self.connected_nodes:
			string += str(connected_node.id) + ", "

		string += "]"

		return string

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
		self.innovation = 0

	def __str__(self):
		string = ""

		string += "IN: " + str(self.in_node_id) + " "
		string += "OUT: " + str(self.out_node_id) + " "
		string += "WEIGHT: " + str(round(self.weight, 3)) + " "
		string += "ENABLED: " + str(self.enabled) + " "
		string += "INNOVATION: " + str(self.innovation)

		return string

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