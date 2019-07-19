import unittest
import sys

sys.path.append('../.')

from neat.neat import Node
from neat.neat import Gene
from neat.neat import Network
from neat.neat import NEAT

class TestNetwork(Network):
	def __init__(self):
		super().__init__()

		self._add_input_node(1)
		self._add_output_node(2)
		self._set_up_genes()

	def _set_up_genes(self):
		gene1 = Gene(in_node = 1, out_node = 2, weight = 1.0, enabled = True)

		self.set_genes([gene1])

class NEATTest(unittest.TestCase):
	def test__midifiy_connection_status(self):
		neat = NEAT()
		network = TestNetwork()

		self.assertTrue(network.genes[0].enabled)

		neat._midifiy_connection_status(network)

		self.assertFalse(network.genes[0].enabled)

		neat._midifiy_connection_status(network)

		self.assertTrue(network.genes[0].enabled)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], 1.0)

	def test__modify_weights(self):
		neat = NEAT()
		network = TestNetwork()

		self.assertEqual(network.genes[0].weight, 1.0)

		neat._modify_weight(network)

		self.assertTrue(network.genes[0].weight != 1.0)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], network.genes[0].weight)

	def test__set_new_random_weight(self):
		neat = NEAT()
		network = TestNetwork()

		self.assertEqual(network.genes[0].weight, 1.0)

		neat._set_new_random_weight(network)

		self.assertTrue(network.genes[0].weight != 1.0)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], network.genes[0].weight)

	def test__set_new_random_weight_all(self):
		neat = NEAT()
		network = TestNetwork()
		genes = network.genes

		gene1 = Gene(in_node = 1, out_node = 3, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = 1, out_node = 4, weight = 1.0, enabled = True)
		gene3 = Gene(in_node = 1, out_node = 5, weight = 1.0, enabled = True)

		gene4 = Gene(in_node = 3, out_node = 2, weight = 1.0, enabled = True)
		gene5 = Gene(in_node = 4, out_node = 2, weight = 1.0, enabled = True)
		gene6 = Gene(in_node = 5, out_node = 2, weight = 1.0, enabled = True)

		genes.append(gene1)
		genes.append(gene2)
		genes.append(gene3)
		genes.append(gene4)
		genes.append(gene5)
		genes.append(gene6)

		network.set_genes(genes)

		for gene in network.genes:
			self.assertEqual(gene.weight, 1.0)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[4].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[5].in_nodes_weights[0][0].id, 1)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[2].in_nodes_weights[1][0].id, 3)
		self.assertEqual(network.nodes[2].in_nodes_weights[2][0].id, 4)
		self.assertEqual(network.nodes[2].in_nodes_weights[3][0].id, 5)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][1], network.genes[1].weight)
		self.assertEqual(network.nodes[4].in_nodes_weights[0][1], network.genes[2].weight)
		self.assertEqual(network.nodes[5].in_nodes_weights[0][1], network.genes[3].weight)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], network.genes[0].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[1][1], network.genes[4].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[2][1], network.genes[5].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[3][1], network.genes[6].weight)

		neat._set_new_random_weight_all(network)

		for gene in network.genes:
			self.assertTrue(gene.weight != 1.0)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[4].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[5].in_nodes_weights[0][0].id, 1)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[2].in_nodes_weights[1][0].id, 3)
		self.assertEqual(network.nodes[2].in_nodes_weights[2][0].id, 4)
		self.assertEqual(network.nodes[2].in_nodes_weights[3][0].id, 5)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][1], network.genes[1].weight)
		self.assertEqual(network.nodes[4].in_nodes_weights[0][1], network.genes[2].weight)
		self.assertEqual(network.nodes[5].in_nodes_weights[0][1], network.genes[3].weight)

		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], network.genes[0].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[1][1], network.genes[4].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[2][1], network.genes[5].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[3][1], network.genes[6].weight)

		self.assertTrue(network.nodes[3].in_nodes_weights[0][1] != 1.0)
		self.assertTrue(network.nodes[4].in_nodes_weights[0][1] != 1.0)
		self.assertTrue(network.nodes[5].in_nodes_weights[0][1] != 1.0)

		self.assertTrue(network.nodes[2].in_nodes_weights[0][1] != 1.0)
		self.assertTrue(network.nodes[2].in_nodes_weights[1][1] != 1.0)
		self.assertTrue(network.nodes[2].in_nodes_weights[2][1] != 1.0)
		self.assertTrue(network.nodes[2].in_nodes_weights[3][1] != 1.0)

	def test__get_net_network_node_id(self):
		neat = NEAT()
		network = TestNetwork()
		genes = network.genes

		gene1 = Gene(in_node = 1, out_node = 3, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = 1, out_node = 4, weight = 1.0, enabled = True)
		gene3 = Gene(in_node = 1, out_node = 5, weight = 1.0, enabled = True)

		gene4 = Gene(in_node = 3, out_node = 2, weight = 1.0, enabled = True)
		gene5 = Gene(in_node = 4, out_node = 2, weight = 1.0, enabled = True)
		gene6 = Gene(in_node = 5, out_node = 2, weight = 1.0, enabled = True)

		genes.append(gene1)
		genes.append(gene2)
		genes.append(gene3)
		genes.append(gene4)
		genes.append(gene5)
		genes.append(gene6)

		network.set_genes(genes)

		new_id = neat._get_net_network_node_id(network)

		self.assertEqual(new_id, 6)

	def test__generate_new_node(self):
		neat = NEAT()
		network = TestNetwork()

		self.assertEqual(len(network.nodes), 3)

		neat._generate_new_node(network)

		self.assertEqual(len(network.nodes), 4)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[2].in_nodes_weights[0][0].id, 3)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][1], network.genes[1].weight)
		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], network.genes[2].weight)

		self.assertEqual(network.nodes[3].in_nodes_weights[0][1], 1.0)
		self.assertEqual(network.nodes[2].in_nodes_weights[0][1], 1.0)

	def test__generate_new_connection(self):
		neat = NEAT()
		network = TestNetwork()

		neat._generate_new_connection(network)