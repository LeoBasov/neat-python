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
		self._add_input_node(2)

		self._add_output_node(3)

		self._set_up_genes()

	def _set_up_genes(self):
		gene1 = Gene(in_node = 1, out_node = 4, weight = 1.0, enabled = True)
		gene2 = Gene(in_node = 2, out_node = 4, weight = 3.0, enabled = True)
		gene3 = Gene(in_node = 4, out_node = 3, weight = 7.0, enabled = True)

		self.set_genes([gene1, gene2, gene3])

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
		network = TestNetwork()
		test_inuput_values_node_ids = ((2, 1), (3, 2))

		network.execute(test_inuput_values_node_ids)

		self.assertEqual(network.nodes[4].value, 11)
		self.assertEqual(network.nodes[3].value, 77)

class NEATTest(unittest.TestCase):
	def test_neat(self):
		neat = NEAT()
		network = TestNetwork()

		new_network = neat.mutate(network)