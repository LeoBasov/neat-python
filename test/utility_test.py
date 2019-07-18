import unittest
import sys

sys.path.append('../.')

import neat.utility as utility

class UtilityTest(unittest.TestCase):
	def test_sigmoid(self):
		value_min = -100.0
		value_max =  100.0
		value_mean = 0.0

		self.assertTrue(utility.sigmoid(value_min) < 1.0e-10)
		self.assertEqual(utility.sigmoid(value_mean), 0.5)
		self.assertTrue(utility.sigmoid(value_max) > 0.9999999)