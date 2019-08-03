import unittest
import sys

sys.path.append('../.')

from neat.network import Node
from neat.network import BiasNode
from neat.network import InputNode
from neat.network import HiddenNode
from neat.network import OutputNode
from neat.network import Network
from neat.network import Connection

from neat.genome import Genome
from neat.genome import Gene

class TestGenome(Genome):
	def __init__(self):
		super().__init__()

		self.allocate_hidden_nodes(1)

		input_node_id_1 = self.add_input_node()
		input_node_id_2 = self.add_input_node()

		output_node_id = self.add_output_node()

		genes = []

		genes.append(Gene(0              , output_node_id))
		genes.append(Gene(input_node_id_1, output_node_id))
		genes.append(Gene(input_node_id_2, output_node_id))

		self.set_genes(genes)
		self.allocate_genes(2)

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

class XORNetwork(Network):
	def __init__(self):
		super().__init__()

		genome = Genome()
		genes = []

		genome.allocate_hidden_nodes(2)

		self.bias_node_id = 0

		self.input_node_id_1 = genome.add_input_node()
		self.input_node_id_2 = genome.add_input_node()

		self.hidden_node_id_1 = genome.add_hidden_node()
		self.hidden_node_id_2 = genome.add_hidden_node()

		self.output_node_id = genome.add_output_node()

		genes.append(Gene(in_node_id = self.bias_node_id, out_node_id = self.hidden_node_id_1, weight = -2.32161229))
		genes.append(Gene(in_node_id = self.bias_node_id, out_node_id = self.hidden_node_id_2, weight = -5.2368337))
		genes.append(Gene(in_node_id = self.bias_node_id, out_node_id = self.output_node_id, weight = -3.13762134))

		genes.append(Gene(in_node_id = self.input_node_id_1, out_node_id = self.hidden_node_id_1, weight = 5.70223616))
		genes.append(Gene(in_node_id = self.input_node_id_1, out_node_id = self.hidden_node_id_2, weight = 3.42762429))

		genes.append(Gene(in_node_id = self.input_node_id_2, out_node_id = self.hidden_node_id_1, weight = 5.73141813))
		genes.append(Gene(in_node_id = self.input_node_id_2, out_node_id = self.hidden_node_id_2, weight = 3.4327536))

		genes.append(Gene(in_node_id = self.hidden_node_id_1, out_node_id = self.output_node_id, weight = 7.05553511))
		genes.append(Gene(in_node_id = self.hidden_node_id_2, out_node_id = self.output_node_id, weight = -7.68450564))

		genome.set_genes(genes)

		self.set_up(genome)

		
class NetworkSlimTest(unittest.TestCase):
	def test_network_initialization(self):
		network = Network()

	def test_execute(self):
		Genome.reset()
		network = XORNetwork()

		value_11 = 1
		value_12 = 0

		value_21 = 0
		value_22 = 1

		value_31 = 0
		value_32 = 0

		value_41 = 1
		value_42 = 1

		input_values_node_ids1 = [(value_11, network.input_node_id_1), (value_12, network.input_node_id_2)]
		input_values_node_ids2 = [(value_21, network.input_node_id_1), (value_22, network.input_node_id_2)]
		input_values_node_ids3 = [(value_31, network.input_node_id_1), (value_32, network.input_node_id_2)]
		input_values_node_ids4 = [(value_41, network.input_node_id_1), (value_42, network.input_node_id_2)]

		results1 = network.execute(input_values_node_ids1)
		results2 = network.execute(input_values_node_ids2)
		results3 = network.execute(input_values_node_ids3)
		results4 = network.execute(input_values_node_ids4)

		self.assertEqual(int(value_11 != value_12), round(results1[5]))
		self.assertEqual(int(value_21 != value_22), round(results2[5]))
		self.assertEqual(int(value_31 != value_32), round(results3[5]))
		self.assertEqual(int(value_41 != value_42), round(results4[5]))

	def test_mate(self):
		Genome.reset()

		for pair in Genome.GENE_INNOVATION_PAIRS:
			print(pair[0])
			print(pair[1])

		genome_parent_1 = TestGenome()
		genome_parent_2 = TestGenome()
		genome_child = TestGenome()

		network_parent_1 = Network(genome_parent_1)
		network_parent_2 = Network(genome_parent_2)
		network_child = Network(genome_child)

		genome_parent_1.add_new_node(0)
		genome_parent_2.add_new_node(1)
		genome_child.add_new_node(2)

		Genome.mate(genome_parent_1, genome_parent_2, genome_child)

		network_child.set_up(genome_child)

		for i in range(-1, -4, -1):
			self.assertTrue(network_child.genome.genes[i].enabled)
			self.assertTrue(network_child.genome.genes[i].used)

		
		self.assertEqual(network_child.genome.genes[3].innovation, 1)
		self.assertEqual(network_child.genome.genes[4].innovation, 2)
		self.assertEqual(network_child.genome.genes[5].innovation, 3)

class NodeSlimTest(unittest.TestCase):
	def test_node_initialization(self):
		node1 = BiasNode()
		node2 = InputNode(1)
		node3 = HiddenNode(2)
		node4 = OutputNode(3)