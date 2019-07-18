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

class NEATTest(unittest.TestCase):
	def test_new_node_neat(self):
		neat = NEAT()
		network = TestNetwork()

		neat.new_node_prob = 1.0

		new_network = neat.mutate(network)

		self.assertEqual(len(network.nodes), 5)
		self.assertEqual(len(new_network.nodes), 6)

	def test_new_connection_neat(self):
		pass
		"""neat = NEAT()
		network = TestNetwork()

		neat.new_connection_prob = 1.0

		new_network = neat.mutate(network)

		self.assertEqual(len(network.genes), 3)
		self.assertEqual(len(new_network.genes), 4)"""