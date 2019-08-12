#!/usr/bin/env python3

from xor_neat import XOR_NEAT
from neat.visual import Visualizer

def main():
	neat = XOR_NEAT()
	visualizer = Visualizer(neat.file_dir)
	parameters = {}

	parameters["number_networks"] = 150
	parameters["number_itterations"] = 1000
	parameters["number_sub_cycles"] = 1

	parameters["test_case_name"] = "XOR NETWORK EVOLUTION"
	parameters["test_case_specifics"] = ["Evolution of XOR network starting with minimal configuration", "Two input and one output node"]

	parameters["number_hidden_nodes"] = 1
	parameters["number_genes"] = 4

	neat.start(**parameters)

	visualizer.plot_fitness('fitness.csv', 'fitness.png')
	visualizer.plot_species('species.png')

if __name__ == '__main__':
	main()
