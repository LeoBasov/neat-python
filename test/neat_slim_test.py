import unittest
import sys
import copy

sys.path.append('../.')

from neat.neat_slim import NEAT
from neat.network_slim import Network
from neat.genome import Genome
from neat.genome import Gene

class TestGenome1(Genome):
	def __init__(self):
		super().__init__()
		genes = []

		self.input_node_id1 = self.add_input_node()
		self.input_node_id2 = self.add_input_node()

		self.output_node_id1 = self.add_output_node()

		self.set_genes(genes)

class TestGenome2(Genome):
	def __init__(self):
		super().__init__()
		genes = []

		self.input_node_id1 = self.add_input_node()
		self.input_node_id2 = self.add_input_node()

		self.output_node_id1 = self.add_output_node()

		genes.append(Gene(self.input_node_id1, self.output_node_id1, weight = 5.0))

		self.set_genes(genes)

class NEATTest(unittest.TestCase):
	def test_add_new_connection(self):
		neat = NEAT()
		genome = TestGenome1()
		new_genome = TestGenome1()
		network = Network(genome)

		neat.add_new_connection(new_genome)

		new_network = Network(new_genome)

	def test_add_new_node(self):
		neat = NEAT()
		genome = TestGenome2()
		new_genome = copy.deepcopy(genome)
		network = Network(genome)

		neat.add_new_node(new_genome)

		new_network = Network(new_genome)

		self.assertEqual(network.genome.genes[0].weight, new_network.genome.genes[-1].weight)
		self.assertEqual(new_network.genome.genes[-2].weight, 1.0)

		self.assertEqual(network.genome.genes[0].in_node_id, new_network.genome.genes[-2].in_node_id)
		self.assertEqual(network.genome.genes[0].out_node_id, new_network.genome.genes[-1].out_node_id)

		self.assertFalse(new_network.genome.genes[0].enabled)

	def test_modify_connection_weight(self):
		neat = NEAT()
		genome = TestGenome2()
		new_genome = TestGenome2()
		network = Network(genome)

		neat.modify_connection_weight(new_genome)

		new_network = Network(new_genome)

		self.assertTrue(network.genome.genes[0].weight != new_network.genome.genes[0].weight)

	def test_change_connection_status(self):
		neat = NEAT()
		genome = TestGenome2()
		new_genome = TestGenome2()
		network = Network(genome)

		neat.change_connection_status(new_genome)

		new_network = Network(new_genome)

		self.assertFalse(new_network.genome.genes[0].enabled)

	def test_mutate(self):
		neat = NEAT()
		genome = TestGenome2()
		network = Network(genome)

		neat.probabilities[1].value = 1.0

		new_network = neat.mutate(network)

		self.assertEqual(network.genome.genes[0].weight, new_network.genome.genes[-1].weight)
		self.assertEqual(new_network.genome.genes[-2].weight, 1.0)

		self.assertEqual(network.genome.genes[0].in_node_id, new_network.genome.genes[-2].in_node_id)
		self.assertEqual(network.genome.genes[0].out_node_id, new_network.genome.genes[-1].out_node_id)

		self.assertFalse(new_network.genome.genes[0].enabled)