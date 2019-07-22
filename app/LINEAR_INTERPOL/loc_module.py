import sys
import random
import math
import enum

sys.path.append('../../.')

from neat.neat import Node
from neat.neat import Gene
from neat.neat import Network
from neat.neat import NEAT

class InputNodeType(enum.Enum):
	#Imput node type
	L_VALUE = 1
	R_VALUE = 2

	MAX_ID = 3

class LinInterpolNetwork(Network):
	def __init__(self, discretisation):
		super().__init__()
		self.__output_ids = discretisation*[0]

		self.set_up(discretisation)
		self.__set_up_genes()

	def set_up(self, discretisation):
		genes = []

		self._add_input_node(InputNodeType.L_VALUE.value)
		self._add_input_node(InputNodeType.R_VALUE.value)

		for i in range(discretisation):
			new_id = InputNodeType.MAX_ID.value + i

			self._add_output_node(new_id)
			self.__output_ids.append(new_id)

	def __set_up_genes(self):
		bias_node_id = 0
		genes = []

		for out_node_id in self.output_node_ids:
			for in_node_id in self.input_node_ids:
				gene = Gene(in_node = in_node_id, out_node = out_node_id, weight = self.__get_random_weight(), enabled = True)
				genes.append(gene)

			gene = Gene(in_node = bias_node_id, out_node = out_node_id, weight = self.__get_random_weight(), enabled = True)
			genes.append(gene)

		self.set_genes(genes)

	def __get_random_weight(self):
		return  10 - 20.0*random.random()