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
import copy

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

	def find_max_innovation(genom1, genom2):
		max_innovation = 0

		for gene in genom1.genes:
			if gene.innovation > max_innovation:
				max_innovation = gene.innovation

		for gene in genom2.genes:
			if gene.innovation > max_innovation:
				max_innovation = gene.innovation

		return max_innovation

	def set_up_base_lists(genom1, genom2):
		base_list = []

		for i in range(min(len(genom1.genes), len(genom2.genes))):
			if genom1.genes[i].innovation == 0:
				base_list.append((genom1.genes[i], genom2.genes[i]))
			else:
				break

		return base_list

	def set_up_innovative_lists(genom1, genom2, max_innovation):
		lists = max_innovation*[None]
		list1 = max_innovation*[None]
		list2 = max_innovation*[None]

		for gene in genom1.genes:
			if gene.innovation > 0:
				list1[gene.innovation - 1] = gene

		for gene in genom2.genes:
			if gene.innovation > 0:
				list2[gene.innovation - 1] = gene

		for i in range(max_innovation):
			lists[i] = (list1[i], list2[i])

		return lists

	def set_up_lists(genom1, genom2):
		max_innovation = Genome.find_max_innovation(genom1, genom2)
		base_lists = Genome.set_up_base_lists(genom1, genom2)
		innovative_lists = Genome.set_up_innovative_lists(genom1, genom2, max_innovation)

		return base_lists + innovative_lists

	def mate(lists, nodes):
		genome = Genome()
		genes = []

		for elem in lists:
			matches = [x for x in elem if x]

			if len(matches) == 2:
				in_node_id = matches[0].in_node_id
				out_node_id = matches[0].out_node_id
				weigth = 0.5*(matches[0].weight + matches[1].weight)
				enabled = matches[0].enabled and matches[1].enabled

				genes.append(Gene(in_node_id, out_node_id, weigth, enabled))
			elif len(matches) == 1:
				genes.append(matches[0])

		genome.set_nodes(nodes)
		genome.set_genes(genes)

		return genome

	def __init__(self):
		self.nodes = [BiasNode()]
		self.output_nodes_ids = []
		self.unused_nodes_ids = []
		self.genes = []

	def allocate_hidden_nodes(self, number):
		for _ in range(number):
			node_id = len(self.nodes)
			node = HiddenNode(node_id, used = False)

			self.nodes.append(node)
			self.unused_nodes_ids.append(node_id)

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

	def set_nodes(self, nodes):
		for node in nodes:
			if node.type == NodeType.OUTPUT:
				self.output_nodes_ids.append(node.id)
			elif node.type == NodeType.BIAS:
				continue

			self.nodes.append(node)

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
	def __init__(self, node_id, node_type, used = True):
		self.id = node_id
		self.type = node_type
		self.level = 0
		self.connected_nodes = []
		self.used = used

	def __str__(self):
		string = ""

		string += "ID: " + str(self.id) + " "
		string += "TYPE: " + str(self.type.name) + " "
		string += "LEVEL: " + str(self.level) + " "
		string += "USED: " + str(self.used) + " "

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
	def __init__(self, node_id, node_type = NodeType.HIDDEN, used = True):
		super().__init__(node_id, node_type, used)

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
	def __init__(self, in_node_id = None, out_node_id = None, weight = 1.0, enabled = True):
		self.in_node_id = in_node_id
		self.out_node_id = out_node_id
		self.weight = weight
		self.enabled = enabled and (in_node_id != None) and (out_node_id != None)
		self.innovation = 0
		self.used = (in_node_id != None) and (out_node_id != None)

	def __str__(self):
		string = ""

		string += "IN: " + str(self.in_node_id) + " "
		string += "OUT: " + str(self.out_node_id) + " "
		string += "WEIGHT: " + str(round(self.weight, 3)) + " "
		string += "ENABLED: " + str(self.enabled) + " "
		string += "INNOVATION: " + str(self.innovation) + " "
		string += "USED: " + str(self.used)

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