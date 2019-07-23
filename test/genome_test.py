import unittest
import sys

sys.path.append('../.')

from neat.genome import BiasNode
from neat.genome import HiddenNode
from neat.genome import NodeType as Type
from neat.genome import Error
from neat.genome import Gene

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