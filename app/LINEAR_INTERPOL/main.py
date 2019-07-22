#!/usr/bin/env python3

from loc_module import LinInterpolNetwork as lin

#Simulation parameters
NUMBER_NETWORKS = 10
NUMBER_ITTERATIONS = 10
L_VALUE = 10
R_VALUE = 10
DISCRITISATION = 10

NETWORKS = []

def main():
	print_header()
	print_set_up()

	set_up_networks()

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

def set_up_networks():
	for _ in range(NUMBER_NETWORKS):
		network = lin(DISCRITISATION)
		NETWORKS.append(network)

if __name__ == '__main__':
	main()