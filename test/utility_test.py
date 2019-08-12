import unittest
import sys

sys.path.append('../.')

import neat.utility as utility

class UtilityTest(unittest.TestCase):
	def test_modified_sigmoid(self):
		value_min = -sys.float_info.max
		value_max =  sys.float_info.max
		value_mean = 0.0

		self.assertTrue(utility.modified_sigmoid(value_min) == 0.0)
		self.assertEqual(utility.modified_sigmoid(value_mean), 0.5)
		self.assertTrue(utility.modified_sigmoid(value_max) == 1.0)

	def test_linear_interpol(self):
		x_0 = 0.0
		y_0 = 1.0
		x_1 = 1.0
		y_1 = 3.0

		result_1 = utility.linear_interpol(x_0, y_0, x_1, y_1, 0.5)
		result_2 = utility.linear_interpol(x_0, y_0, x_1, y_1, 1.5)

		self.assertEqual(2.0, result_1)
		self.assertEqual(4.0, result_2)
