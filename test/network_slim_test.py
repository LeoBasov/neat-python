import unittest
import sys

sys.path.append('../.')

from neat.network_slim import Node
from neat.network_slim import BiasNode
from neat.network_slim import InputNode
from neat.network_slim import HiddenNode
from neat.network_slim import OutputNode
from neat.network_slim import Network
from neat.network_slim import Connection

from neat.genome import Genome
from neat.genome import Gene

class TestNetwork(Network):
	def __init__(self):
		super().__init__()

		genome = Genome()
		genes = []

		self.input_node_id_1 = genome.add_input_node()
		self.input_node_id_2 = genome.add_input_node()

		self.hidden_node_id_1 = genome.add_hidden_node()

		self.output_node_id_1 = genome.add_output_node()
		self.output_node_id_2 = genome.add_output_node()

		genes.append(Gene(self.input_node_id_1, self.hidden_node_id_1))
		genes.append(Gene(self.input_node_id_2, self.hidden_node_id_1))

		genes.append(Gene(self.hidden_node_id_1, self.output_node_id_1))
		genes.append(Gene(self.hidden_node_id_1, self.output_node_id_2))

		genome.set_genes(genes)

		self.set_up(genome)

class NetworkSlimTest(unittest.TestCase):
	def test_network_initialization(self):
		network = Network()

	def test_execute(self):
		network = TestNetwork()

class NodeSlimTest(unittest.TestCase):
	def test_node_initialization(self):
		node1 = BiasNode()
		node2 = InputNode(1)
		node3 = HiddenNode(2)
		node4 = OutputNode(3)