import unittest
import sys

sys.path.append('../.')

import neat.utility as utility

class UtilityTest(unittest.TestCase):
	def test_sigmoid(self):
		value_min = -sys.float_info.max
		value_max =  sys.float_info.max
		value_mean = 0.0

		self.assertTrue(utility.sigmoid(value_min) == 0.0)
		self.assertEqual(utility.sigmoid(value_mean), 0.5)
		self.assertTrue(utility.sigmoid(value_max) == 1.0)