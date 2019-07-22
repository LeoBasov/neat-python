#!/usr/bin/env python3

import sys
import random
import math

sys.path.append('../../.')

from neat.neat import Node
from neat.neat import Gene
from neat.neat import Network
from neat.neat import NEAT

#Simulation parameters
NUMBER_NETWORKS = 10
NUMBER_ITTERATIONS = 10
L_VALUE = 10
R_VALUE = 10
DISCRITISATION = 10

def main():
	print_header()
	print_set_up()

	print_footer()

def print_header():
	print(80*"=")
	print("NEAT - NeuroEvolution of Augmenting Topologies")
	print("(c) 2019, Leo Basov")
	print(80*"-")
	print("LINEAR INTERPOLATION TEST CASE")
	print("Interpolation length: 1")
	print(80*"-")

def print_set_up():
	print("NUMBER_NETWORKS", NUMBER_NETWORKS)
	print("NUMBER_ITTERATIONS", NUMBER_ITTERATIONS)
	print("L_VALUE", L_VALUE)
	print("R_VALUE", R_VALUE)

def print_footer():
	print(80*"-")
	print("Execution finished")
	print(80*"=")

if __name__ == '__main__':
	main()