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
	GENE_INNOVATION_PAIRS = []

	def reset():
		Genome.GENE_INNOVATION_PAIRS = []

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

	def mate(genome_parent_1, genome_parent_2, genome_child):
		lists = Genome.set_up_lists(genome_parent_1, genome_parent_2)
		nodes = genome_parent_1.nodes if len(genome_parent_1.nodes) > len(genome_parent_2.nodes) else genome_parent_2.nodes
		genes = []
		used_genes = []
		unused_genes = []

		for elem in lists:
			matches = [x for x in elem if x]

			if len(matches) == 2:
				in_node_id = matches[0].in_node_id
				out_node_id = matches[0].out_node_id
				weigth = 0.5*(matches[0].weight + matches[1].weight)
				enabled = matches[0].enabled and matches[1].enabled
				innovation = matches[0].innovation

				genes.append(Gene(in_node_id, out_node_id, weigth, enabled, innovation))
			elif len(matches) == 1:
				genes.append(matches[0])

		for gene in genes:
			if gene.used:
				used_genes.append(gene)
			else:
				unused_genes.append(gene)

		genes = used_genes + unused_genes

		genome_child.unused_gene_index = len(used_genes)
		genome_child.unused_nodes_current_id = max(genome_parent_1.unused_nodes_current_id, genome_parent_2.unused_nodes_current_id)
		genome_child.set_nodes(nodes)
		genome_child.set_genes(genes)

	def __init__(self):
		self.nodes = [BiasNode()]
		self.output_nodes_ids = []
		self.unused_nodes_ids = []
		self.unused_nodes_current_id = 0
		self.genes = []
		self.unused_gene_index = 0

	def __str__(self):
		string = ''

		string += 80*'-' + '\n'
		string += 'NODES\n'
		string += 80*'-' + '\n'

		for node in self.nodes:
			string += node.__str__() + '\n'

		string += 80*'-' + '\n'
		string += 'GENES\n'
		string += 80*'-' + '\n'

		for gene in self.genes:
			string += gene.__str__() + '\n'

		return string


	def allocate_hidden_nodes(self, number):
		for _ in range(number):
			node_id = len(self.nodes)
			node = HiddenNode(node_id)

			self.nodes.append(node)
			self.unused_nodes_ids.append(node_id)

	def allocate_genes(self, number):
		"""Function to allocate genes for later use in mutation

		This function allocates genes that will be later used and activated during the mutation process.
		This function muss be called after the set_genes function.

		Parameters
		----------
		number : int
			number of allocated genes

		Returns
		-------
		None

		"""

		self.unused_gene_index = len(self.genes)

		for _ in range(number):
			gene = Gene()

			self.genes.append(gene)

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
		node_id = self.unused_nodes_ids[self.unused_nodes_current_id]
		
		self.unused_nodes_current_id += 1

		return node_id

	def set_nodes(self, nodes):
		self.nodes = [BiasNode()]

		for node in nodes:
			node.connected_nodes = []

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

				if gene.used and gene.enabled:
					self.nodes[gene.out_node_id].connected_nodes.append(self.nodes[gene.in_node_id])
			else:
				raise Error("Genome.set_genes", "Gene defined twice")

		self.update_gene_innovation_pairs()
		self.update_levels()
		self.update_unused_nodes_gene_index()

	def update_gene_innovation_pairs(self):
		if len(Genome.GENE_INNOVATION_PAIRS) == 0:
			for gene in self.genes:
				Genome.GENE_INNOVATION_PAIRS.append((gene, 0))

	def update_unused_nodes_gene_index(self):
		self.unused_gene_index = 0

		for gene in self.genes:
			if gene.used:
				self.unused_gene_index += 1
			else:
				break

	def add_new_connection(self, in_node_id, out_node_id, weight = 1.0, enabled = True):
		if self.connection_possible(in_node_id, out_node_id):
			gene = self.genes[self.unused_gene_index]

			gene.in_node_id = in_node_id
			gene.out_node_id = out_node_id
			gene.weight = weight
			gene.enabled = enabled
			gene.used = True

			self.unused_gene_index += 1
			self.add_gene(gene)

			return True
		else:
			return False

	def add_new_connection_no_check(self, in_node_id, out_node_id, weight = 1.0, enabled = True):
		gene = self.genes[self.unused_gene_index]

		gene.in_node_id = in_node_id
		gene.out_node_id = out_node_id
		gene.weight = weight
		gene.enabled = enabled
		gene.used = True

		self.unused_gene_index += 1
		self.add_gene(gene)

	def connection_possible(self, in_node_id, out_node_id):
		used_nodes = (in_node_id not in self.unused_nodes_ids[self.unused_nodes_current_id:]) and (out_node_id not in self.unused_nodes_ids[self.unused_nodes_current_id:])
		no_in_output = self.nodes[in_node_id].type != NodeType.OUTPUT
		no_out_input = self.nodes[out_node_id].type != NodeType.INPUT
		no_level_conflict = self.nodes[in_node_id].level < self.nodes[out_node_id].level
		no_existence = True

		for gene in self.genes:
			if gene.connection_exists(in_node_id, out_node_id):
				no_existence = False
				break

		return used_nodes and no_in_output and no_out_input and no_level_conflict and no_existence

	def add_gene(self, gene):
		self.check_gene(gene)
		self.nodes[gene.out_node_id].connected_nodes.append(self.nodes[gene.in_node_id])

	def add_new_node(self, gene_id):
		if self.genes[gene_id].enabled and self.genes[gene_id].used and ((self.unused_gene_index + 1) < len(self.genes)):
			self.genes[gene_id].enabled = False
			in_node_id = self.genes[gene_id].in_node_id
			out_node_id = self.genes[gene_id].out_node_id
			new_node_id = self.add_hidden_node()
			weight = self.genes[gene_id].weight

			gene1 = self.genes[self.unused_gene_index]
			gene2 = self.genes[self.unused_gene_index + 1]

			gene1.in_node_id = in_node_id
			gene2.in_node_id = new_node_id

			gene1.out_node_id = new_node_id
			gene2.out_node_id = out_node_id

			gene1.weight = 1.0
			gene2.weight = weight

			gene1.enabled = True
			gene2.enabled = True

			gene1.used = True
			gene2.used = True

			self.unused_gene_index += 2
			
			self.add_gene(gene1)
			self.add_gene(gene2)

			self.update_levels()

	def update_levels(self):
		for node_id in self.output_nodes_ids:
			self.nodes[node_id].update_level()

	def check_gene(self, gene):
		for pair in Genome.GENE_INNOVATION_PAIRS:
			if pair[0].connection_exists(gene.in_node_id, gene.out_node_id):
				gene.innovation = pair[1]
				return

		Genome.GENE_INNOVATION_PAIRS.append((gene, Genome.GENE_INNOVATION_PAIRS[-1][1] + 1))
		gene.innovation = Genome.GENE_INNOVATION_PAIRS[-1][1]

class Node:
	def __init__(self, node_id, node_type, used = True):
		self.id = node_id
		self.type = node_type
		self.level = 0
		self.connected_nodes = []

	def __str__(self):
		string = ""

		string += "ID: {:3d} ".format(self.id)
		string += "TYPE: {:6s} ".format(self.type.name)
		string += "LEVEL: {:3d} ".format(self.level)

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
	def __init__(self, in_node_id = None, out_node_id = None, weight = 1.0, enabled = True, innovation = 0):
		self.in_node_id = in_node_id
		self.out_node_id = out_node_id
		self.weight = weight
		self.enabled = enabled and (in_node_id != None) and (out_node_id != None)
		self.innovation = innovation
		self.used = (in_node_id != None) and (out_node_id != None)

	def __str__(self):
		string = ""

		string += f"IN: {'{:4d}'.format(self.in_node_id) if self.in_node_id != None else 'None'} "
		string += f"OUT: {'{:4d}'.format(self.out_node_id) if self.out_node_id else 'None'} "
		string += f"WEIGHT: {'{:6.3f}'.format(self.weight)} "
		string += f"ENABLED: {'True ' if self.enabled else 'False'} "
		string += "INNOVATION: {:3d} ".format(self.innovation)
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