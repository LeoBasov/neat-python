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

	def test__change_weights(self):
		neat = NEAT()
		network = TestNetwork()

		self.assertEqual(network.genes[0].weight, 1.0)

		neat._change_weights(network)

		self.assertTrue(network.genes[0].weight != 1.0)