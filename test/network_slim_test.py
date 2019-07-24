import unittest
import sys

sys.path.append('../.')

from neat.network_slim import Node
from neat.network_slim import Network

class NetworkSlimTest(unittest.TestCase):
	def test_network_initialization(self):
		network = Network()