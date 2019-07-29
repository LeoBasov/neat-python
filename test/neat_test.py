import unittest
import sys
import copy

sys.path.append('../.')

from neat.neat import Mutator
from neat.neat import Species
from neat.network import Network
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

		self.allocate_hidden_nodes(1)

		self.input_node_id1 = self.add_input_node()
		self.input_node_id2 = self.add_input_node()

		self.output_node_id1 = self.add_output_node()

		genes.append(Gene(self.input_node_id1, self.output_node_id1, weight = 5.0))

		self.set_genes(genes)
		self.allocate_genes(2)

class TestGenome3(Genome):
	def __init__(self):
		super().__init__()
		genes = []

		self.allocate_hidden_nodes(1)

		self.input_node_id1 = self.add_input_node()
		self.input_node_id2 = self.add_input_node()

		self.output_node_id1 = self.add_output_node()

		genes.append(Gene(self.input_node_id1, self.output_node_id1, weight = 5.0))

		self.set_genes(genes)

class MutatorTest(unittest.TestCase):
	def test_add_new_connection(self):
		neat = Mutator()
		genome = TestGenome2()

		neat.add_new_connection(genome)

		network = Network(genome)

	def test_add_new_node(self):
		neat = Mutator()
		genome = TestGenome2()
		new_genome = copy.deepcopy(genome)
		network = Network(genome)

		neat.add_new_node(new_genome)

		while (not new_genome.genes[-1].used):
			neat.add_new_node(new_genome)

		new_network = Network(new_genome)

		self.assertEqual(network.genome.genes[0].weight, new_network.genome.genes[-1].weight)
		self.assertEqual(new_network.genome.genes[-2].weight, 1.0)

		self.assertEqual(network.genome.genes[0].in_node_id, new_network.genome.genes[-2].in_node_id)
		self.assertEqual(network.genome.genes[0].out_node_id, new_network.genome.genes[-1].out_node_id)

		self.assertFalse(new_network.genome.genes[0].enabled)

	def test_modify_connection_weight(self):
		neat = Mutator()
		genome = TestGenome3()
		new_genome = TestGenome3()
		network = Network(genome)

		neat.modify_connection_weight(new_genome)

		new_network = Network(new_genome)

		self.assertTrue(network.genome.genes[0].weight != new_network.genome.genes[0].weight)

	def test_change_connection_status(self):
		neat = Mutator()
		genome = TestGenome3()
		new_genome = TestGenome3()
		network = Network(genome)

		neat.change_connection_status(new_genome)

		new_network = Network(new_genome)

		self.assertFalse(new_network.genome.genes[0].enabled)

	def test_mutate(self):
		neat = Mutator()
		genome = TestGenome2()
		network = Network(genome)
		new_network = Network(genome)

		neat.probabilities[1].value = 1.0

		neat.mutate(new_network)

		self.assertEqual(network.genome.genes[0].weight, new_network.genome.genes[-1].weight)
		self.assertEqual(new_network.genome.genes[-2].weight, 1.0)

		self.assertEqual(network.genome.genes[0].in_node_id, new_network.genome.genes[-2].in_node_id)
		self.assertEqual(network.genome.genes[0].out_node_id, new_network.genome.genes[-1].out_node_id)

		self.assertFalse(new_network.genome.genes[0].enabled)

class SpeciesTest(unittest.TestCase):
	def test_distance(self):
		Genome.reset()

		genome1 = TestGenome3()
		genome2 = TestGenome2()
		species = Species(genome1)

		genome2.add_new_node(0)

		distance1 = species.compare(genome1)
		distance2 = species.compare(genome2)

		self.assertEqual(distance1, 0.0)
		self.assertEqual(distance2, 2.0/3.0)