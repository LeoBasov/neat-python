import unittest
import sys
import random

sys.path.append('../.')

from neat.network import Node
from neat.network import Gene
from neat.network import Network
import neat.utility as utility

class TestNetwork(Network):
	def __init__(self):
		super().__init__()

		self._add_input_node(1)
		self._add_input_node(2)

		self._add_output_node(3)

		self._set_up_genes()

	def _set_up_genes(self):
		gene1 = Gene(in_node = 1, out_node = 4, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = 2, out_node = 4, weight = 3.0, enabled = True)
		gene3 = Gene(in_node = 4, out_node = 3, weight = 7.0, enabled = True)

		self.set_genes([gene1, gene2, gene3])

class XORNetwork(Network):
	def __init__(self):
		super().__init__()

		self._add_input_node(1)
		self._add_input_node(2)

		self._add_output_node(3)

		self._set_up_genes()

	def _set_up_genes(self):
		gene1 = Gene(in_node = 0, out_node = 4, weight = -2.32161229, enabled = True)
		gene2 = Gene(in_node = 0, out_node = 5, weight = -5.2368337, enabled = True)
		gene3 = Gene(in_node = 0, out_node = 3, weight = -3.13762134, enabled = True)

		gene4 = Gene(in_node = 1, out_node = 4, weight = 5.70223616, enabled = True)
		gene5 = Gene(in_node = 1, out_node = 5, weight = 3.42762429, enabled = True)

		gene6 = Gene(in_node = 2, out_node = 4, weight = 5.73141813, enabled = True)
		gene7 = Gene(in_node = 2, out_node = 5, weight = 3.4327536, enabled = True)

		gene8 = Gene(in_node = 4, out_node = 3, weight = 7.05553511, enabled = True)
		gene9 = Gene(in_node = 5, out_node = 3, weight = -7.68450564, enabled = True)

		self.set_genes([gene1, gene2, gene3, gene4, gene5, gene6, gene7, gene8, gene9])

class NetworkTest(unittest.TestCase):
	def test_network_initialization(self):
		network = TestNetwork()
		test_inuput_values_node_ids = ((2, 1), (3, 2))

		#Bias Node test
		self.assertEqual(network.nodes[0].value, 1)
		self.assertEqual(network.nodes[0].id, 0)

		#Input Node test
		self.assertEqual(network.nodes[1].value, 0)
		self.assertEqual(network.nodes[1].id, 1)

		self.assertEqual(network.nodes[2].value, 0)
		self.assertEqual(network.nodes[2].id, 2)

		#Hidden Node test
		self.assertEqual(network.nodes[4].value, 0)
		self.assertEqual(network.nodes[4].id, 4)

		self.assertEqual(len(network.nodes[4].in_nodes_weights), 2)

		self.assertEqual(network.nodes[4].in_nodes_weights[0][0].id, 1)
		self.assertEqual(network.nodes[4].in_nodes_weights[0][1], 1)

		self.assertEqual(network.nodes[4].in_nodes_weights[1][0].id, 2)
		self.assertEqual(network.nodes[4].in_nodes_weights[1][1], 3)

		#Output Node test
		self.assertEqual(network.nodes[3].value, 0)
		self.assertEqual(network.nodes[3].id, 3)
		self.assertEqual(len(network.nodes[3].in_nodes_weights), 1)
		self.assertEqual(network.nodes[3].in_nodes_weights[0][0].id, 4)
		self.assertEqual(network.nodes[3].in_nodes_weights[0][1], 7)

	def test_network_execution(self):
		network = XORNetwork()

		for _ in range(10):
			val1 = round(random.random())
			val2 = round(random.random())

			input_values_node_ids = [(val1, 1), (val2, 2)]

			network.execute(input_values_node_ids)

			self.assertEqual(int(val1 != val2), round(utility.sigmoid(network.nodes[3].value)))