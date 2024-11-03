# Author: Amirmahdi Joudi
# This file includes code for solving a circuit containing resistance, voltage and current sources described in HSPICE
# Class CircuitSolver is the main part of this code
# parse_netlist is a method for reading the netlist, finding resistors, voltagesources, current sources, and unique nodes
# build_g_j_vectors is a method to make g and j matrices
# print_matrices is a method to print g and j matrices
# get_node_index is a method to give proper index for each node
# solve is a method for solving equations using spsolve
# display_results is a method to show the node voltages
# circuit.export_matrices_to_csv is a method to export g and j matrices as csv file (output is large!)

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import csv
import time

class CircuitSolver:
    def __init__(self):
        self.nodes = {} 
        self.resistors = []
        self.current_sources = []
        self.voltage_sources = []
        self.num_nodes = 0
        self.g_matrix = None
        self.j_vector = None
    
    def parse_netlist(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                tokens = line.split()
                if not tokens:
                    continue
                
                if tokens[0][0] == 'R':
                    _, node1, node2, resistance = tokens
                    resistance = float(resistance)
                    self.resistors.append((node1, node2, resistance))
                
                elif tokens[0][0] == 'I':
                    _, node1, node2, current = tokens
                    current = float(current)
                    self.current_sources.append((node1, node2, current))
                
                elif tokens[0][0] == 'V':
                    _, node1, node2, voltage = tokens
                    voltage = float(voltage)
                    self.voltage_sources.append((node1, node2, voltage))

                for node in [node1, node2]:
                    if node != '0' and node not in self.nodes:
                        self.nodes[node] = len(self.nodes) + 1

                self.num_nodes = len(self.nodes)
    
    def build_g_j_vectors(self):
        n = self.num_nodes + len(self.voltage_sources)
        self.g_matrix = lil_matrix((n, n), dtype=np.float64)
        self.j_vector = np.zeros(n)

        for node1, node2, resistance in self.resistors:
            conductance = 1 / resistance
            i, j = self.get_node_index(node1), self.get_node_index(node2)
            if i != -1:
                self.g_matrix[i, i] += conductance
            if j != -1:
                self.g_matrix[j, j] += conductance
            if i != -1 and j != -1:
                self.g_matrix[i, j] -= conductance
                self.g_matrix[j, i] -= conductance
        
        for node1, node2, current in self.current_sources:
            i, j = self.get_node_index(node1), self.get_node_index(node2)
            if i != -1:
                self.j_vector[i] -= current
            if j != -1:
                self.j_vector[j] += current

        for k, (node1, node2, voltage) in enumerate(self.voltage_sources):
            i, j = self.get_node_index(node1), self.get_node_index(node2)
            voltage_index = self.num_nodes + k
            if i != -1:
                self.g_matrix[voltage_index, i] = 1
                self.g_matrix[i, voltage_index] = 1
            if j != -1:
                self.g_matrix[voltage_index, j] = -1
                self.g_matrix[j, voltage_index] = -1
            self.j_vector[voltage_index] = voltage

        self.g_matrix = self.g_matrix.tocsr()

    def print_matrices(self):
        print("G matrix:")
        print(self.g_matrix.toarray())
        print("J matrix:")
        print(self.j_vector)

    def get_node_index(self, node):
        if node == '0':
            return -1
        return self.nodes[node] - 1

    def solve(self):
        voltages = spsolve(self.g_matrix, self.j_vector)
        return voltages[:self.num_nodes]

    def export_matrices_to_csv(self, file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in self.g_matrix.toarray():
                writer.writerow(row)

    def display_results(self, voltages):
        print("Node Voltages:")
        for node, index in self.nodes.items():
            print(f"V({node}) = {voltages[index - 1]:.4f} V")
    
    def get_num_nodes(self):
        return self.num_nodes

# # Example usage:
# circuit = CircuitSolver()
# circuit.parse_netlist('./benchmarks/real-circuit-data/testcase18/netlist.sp')
# circuit.build_g_j_vectors()
# # circuit.print_matrices()
# voltages = circuit.solve()
# circuit.display_results(voltages)
# # circuit.export_matrices_to_csv("./benchmarks/real-circuit-data/testcase1/netlist.csv")  
# print(circuit.get_num_nodes())

repetitions = 20
elapsed_time_total = 0
elapsed_time_netlist_parser = 0
elapsed_time_matrices_builder = 0
elapsed_time_solver = 0
elapsed_time_printing = 0
for i in range(0, repetitions):
    start_time_total = time.perf_counter()
    circuit = CircuitSolver()
    circuit.parse_netlist('./benchmarks/real-circuit-data/testcase18/netlist.sp')
    end_time_netlist_parser = time.perf_counter()
    circuit.build_g_j_vectors()
    end_time_matrices_builder = time.perf_counter()
    # circuit.print_matrices()
    voltages = circuit.solve()
    end_time_solver = time.perf_counter()
    circuit.display_results(voltages)
    # circuit.export_matrices_to_csv("./benchmarks/real-circuit-data/testcase1/netlist.csv")  
    end_time_total = time.perf_counter()
    elapsed_time_total += (end_time_total-start_time_total)
    elapsed_time_netlist_parser += (end_time_netlist_parser-start_time_total)
    elapsed_time_matrices_builder += (end_time_matrices_builder-end_time_netlist_parser)
    elapsed_time_solver += (end_time_solver-end_time_matrices_builder)
    elapsed_time_printing += (end_time_total-end_time_solver)

print(circuit.get_num_nodes())
print(f"Solver Parser Time: {elapsed_time_netlist_parser/repetitions:.6f} seconds")
print(f"Solver Matrices Builder Time: {elapsed_time_matrices_builder/repetitions:.6f} seconds")
print(f"Solver Solving Time: {elapsed_time_solver/repetitions:.6f} seconds")
print(f"Solver Results Printing Time: {elapsed_time_printing/repetitions:.6f} seconds")
print(f"Solver Total Time: {elapsed_time_total/repetitions:.6f} seconds")
