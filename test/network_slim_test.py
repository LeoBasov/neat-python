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

class TestNetwork(Network):
	def __init__(self):
		super().__init__()

		self.nodes.append(BiasNode())

		self.nodes.append(InputNode(1))
		self.nodes.append(InputNode(2))

		self.nodes.append(HiddenNode(3))

		self.nodes.append(OutputNode(4))
		self.nodes.append(OutputNode(5))

		self.nodes[4].connections.append(Connection(self.nodes[3]))
		self.nodes[5].connections.append(Connection(self.nodes[3]))

		self.nodes[3].connections.append(Connection(self.nodes[1]))
		self.nodes[3].connections.append(Connection(self.nodes[2]))

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