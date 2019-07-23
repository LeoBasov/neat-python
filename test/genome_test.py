import unittest
import sys
import random

sys.path.append('../.')

from neat.genome import BiasNode
from neat.genome import HiddenNode
from neat.genome import NodeType as Type
from neat.genome import Error
from neat.genome import Gene
from neat.genome import Genome

class NodeTest(unittest.TestCase):
	def test_init(self):
		bias_node = BiasNode()
		input_node = HiddenNode(1, Type.INPUT)
		hidden_node = HiddenNode(2, Type.HIDDEN)
		output_node = HiddenNode(3, Type.OUTPUT)
		cought = False

		try:
			faulty_node = HiddenNode(0, Type.INPUT)
		except Error as e:
			cought = True
		else:
			self.assertTrue(cought)

class GeneTest(unittest.TestCase):
	def test_connection_exists(self):
		gene = Gene(in_node_id = 0, out_node_id = 1)

		self.assertTrue(gene.connection_exists(1, 0))
		self.assertTrue(gene.connection_exists(0, 1))

		self.assertFalse(gene.connection_exists(1, 1))
		self.assertFalse(gene.connection_exists(0, 0))

class GenomeTest(unittest.TestCase):
	def test_add_node(self):
		genome = Genome()

		input_node_id_1 = genome.add_input_node()
		input_node_id_2 = genome.add_input_node()

		hidden_node_id_1 = genome.add_hidden_node()
		hidden_node_id_2 = genome.add_hidden_node()
		hidden_node_id_3 = genome.add_hidden_node()

		output_node_id_1 = genome.add_output_node()
		output_node_id_2 = genome.add_output_node()

		#Inserted check
		self.assertTrue(input_node_id_1 in range(len(genome.nodes)))
		self.assertTrue(input_node_id_2 in range(len(genome.nodes)))

		self.assertTrue(hidden_node_id_1 in range(len(genome.nodes)))
		self.assertTrue(hidden_node_id_2 in range(len(genome.nodes)))
		self.assertTrue(hidden_node_id_3 in range(len(genome.nodes)))

		self.assertTrue(output_node_id_1 in range(len(genome.nodes)))
		self.assertTrue(output_node_id_2 in range(len(genome.nodes)))

		#Added to output ndoes check
		self.assertFalse(input_node_id_1 in genome.output_nodes_ids)
		self.assertFalse(input_node_id_2 in genome.output_nodes_ids)

		self.assertFalse(hidden_node_id_1 in genome.output_nodes_ids)
		self.assertFalse(hidden_node_id_2 in genome.output_nodes_ids)
		self.assertFalse(hidden_node_id_3 in genome.output_nodes_ids)

		self.assertTrue(output_node_id_1 in genome.output_nodes_ids)
		self.assertTrue(output_node_id_2 in genome.output_nodes_ids)

		#Type set check
		self.assertEqual(genome.nodes[input_node_id_1].type, Type.INPUT)
		self.assertEqual(genome.nodes[input_node_id_2].type, Type.INPUT)

		self.assertEqual(genome.nodes[hidden_node_id_1].type, Type.HIDDEN)
		self.assertEqual(genome.nodes[hidden_node_id_2].type, Type.HIDDEN)
		self.assertEqual(genome.nodes[hidden_node_id_3].type, Type.HIDDEN)

		self.assertEqual(genome.nodes[output_node_id_1].type, Type.OUTPUT)
		self.assertEqual(genome.nodes[output_node_id_2].type, Type.OUTPUT)

		#ID set check
		self.assertEqual(genome.nodes[input_node_id_1].id, input_node_id_1)
		self.assertEqual(genome.nodes[input_node_id_2].id, input_node_id_2)

		self.assertEqual(genome.nodes[hidden_node_id_1].id, hidden_node_id_1)
		self.assertEqual(genome.nodes[hidden_node_id_2].id, hidden_node_id_2)
		self.assertEqual(genome.nodes[hidden_node_id_3].id, hidden_node_id_3)

		self.assertEqual(genome.nodes[output_node_id_1].id, output_node_id_1)
		self.assertEqual(genome.nodes[output_node_id_2].id, output_node_id_2)

	def test_set_genes(self):
		genome = Genome()
		genes = []

		input_node_id_1 = genome.add_input_node()
		input_node_id_2 = genome.add_input_node()

		hidden_node_id_1 = genome.add_hidden_node()

		output_node_id_1 = genome.add_output_node()
		output_node_id_2 = genome.add_output_node()

		genes.append(Gene(input_node_id_1, hidden_node_id_1))
		genes.append(Gene(input_node_id_2, hidden_node_id_1))

		genes.append(Gene(hidden_node_id_1, output_node_id_1))
		genes.append(Gene(hidden_node_id_1, output_node_id_2))

		genome.set_genes((genes))

		self.assertEqual(genome.nodes[output_node_id_1].connected_nodes[0], genome.nodes[hidden_node_id_1])
		self.assertEqual(genome.nodes[output_node_id_2].connected_nodes[0], genome.nodes[hidden_node_id_1])

		self.assertEqual(genome.nodes[hidden_node_id_1].connected_nodes[0], genome.nodes[input_node_id_1])
		self.assertEqual(genome.nodes[hidden_node_id_1].connected_nodes[1], genome.nodes[input_node_id_2])

		self.assertEqual(genome.nodes[input_node_id_1].level, 0)
		self.assertEqual(genome.nodes[input_node_id_2].level, 0)

		self.assertEqual(genome.nodes[hidden_node_id_1].level, 1)

		self.assertEqual(genome.nodes[output_node_id_1].level, 2)
		self.assertEqual(genome.nodes[output_node_id_2].level, 2)

	def test_add_gene(self):
		genome = Genome()
		genes = []

		input_node_id_1 = genome.add_input_node()
		input_node_id_2 = genome.add_input_node()

		hidden_node_id_1 = genome.add_hidden_node()

		output_node_id_1 = genome.add_output_node()
		output_node_id_2 = genome.add_output_node()

		genes.append(Gene(input_node_id_1, hidden_node_id_1))
		genes.append(Gene(input_node_id_2, hidden_node_id_1))

		genes.append(Gene(hidden_node_id_1, output_node_id_1))
		genes.append(Gene(hidden_node_id_1, output_node_id_2))

		genome.set_genes((genes))

		gene = Gene(input_node_id_1, output_node_id_1)

		genome.add_gene(gene)

		self.assertEqual(genome.nodes[output_node_id_1].connected_nodes[1], genome.nodes[input_node_id_1])

	def test_add_new_connection(self):
		genome = Genome()
		genes = []

		input_node_id_1 = genome.add_input_node()
		input_node_id_2 = genome.add_input_node()

		hidden_node_id_1 = genome.add_hidden_node()

		output_node_id_1 = genome.add_output_node()
		output_node_id_2 = genome.add_output_node()

		genes.append(Gene(input_node_id_1, hidden_node_id_1))
		genes.append(Gene(input_node_id_2, hidden_node_id_1))

		genes.append(Gene(hidden_node_id_1, output_node_id_1))
		genes.append(Gene(hidden_node_id_1, output_node_id_2))

		genome.set_genes((genes))

		connected = genome.add_new_connection(input_node_id_1, output_node_id_1)

		self.assertEqual(genome.nodes[output_node_id_1].connected_nodes[1], genome.nodes[input_node_id_1])
		self.assertTrue(connected)

		connected1 = genome.add_new_connection(input_node_id_1, input_node_id_2)
		connected2 = genome.add_new_connection(output_node_id_1, output_node_id_2)
		connected3 = genome.add_new_connection(output_node_id_2, hidden_node_id_1)
		connected4 = genome.add_new_connection(output_node_id_2, input_node_id_1)

		self.assertFalse(connected1)
		self.assertFalse(connected2)
		self.assertFalse(connected3)
		self.assertFalse(connected4)

	def test_add_new_node(self):
		genome = Genome()
		genes = []

		input_node_id_1 = genome.add_input_node()
		input_node_id_2 = genome.add_input_node()

		hidden_node_id_1 = genome.add_hidden_node()

		output_node_id_1 = genome.add_output_node()
		output_node_id_2 = genome.add_output_node()

		genes.append(Gene(input_node_id_1, hidden_node_id_1))
		genes.append(Gene(input_node_id_2, hidden_node_id_1))

		genes.append(Gene(hidden_node_id_1, output_node_id_1))
		genes.append(Gene(hidden_node_id_1, output_node_id_2))

		genome.set_genes((genes))

		gene_id = 0

		genome.add_new_node(gene_id)

		self.assertEqual(genome.nodes[hidden_node_id_1].connected_nodes[0], genome.nodes[input_node_id_1])
		self.assertEqual(genome.nodes[hidden_node_id_1].connected_nodes[1], genome.nodes[input_node_id_2])
		self.assertEqual(genome.nodes[hidden_node_id_1].connected_nodes[2], genome.nodes[6])

		self.assertEqual(genome.nodes[6].connected_nodes[0], genome.nodes[input_node_id_1])