import matplotlib.pyplot as plt
import re
import numpy as np

def parse_file_1(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            # Matches lines like: "+ n1_m1_0_0 = 1.0996"
            match = re.match(r"^\+ (\S+) =\s+([\d.]+)", line)
            if match:
                node = match.group(1)
                voltage = float(match.group(2))
                data[node] = voltage
    return data

def parse_file_2(filename):
    data = {}
    with open(filename, 'r', encoding='utf-16') as file:
        for line in file:
            # Matches lines like: "V(n1_m1_0_0) = 1.0996 V"
            match = re.match(r"^V\((\S+)\) = ([\d.]+) V", line)
            if match:
                node = match.group(1)
                voltage = float(match.group(2))
                data[node] = voltage
    return data

def scatter_plot(data1, data2, max_nodes=500):
    common_nodes = sorted(set(data1.keys()).intersection(data2.keys()))
    # If there are too many nodes, randomly sample them
    if len(common_nodes) > max_nodes:
        common_nodes = np.random.choice(common_nodes, max_nodes, replace=False)

    voltages1 = [data1[node] for node in common_nodes]
    voltages2 = [data2[node] for node in common_nodes]
    plt.figure(figsize=(20, 12))
    plt.scatter(common_nodes, voltages1, color='blue', label='HSPICE Voltage', marker='o', alpha=0.6)
    plt.scatter(common_nodes, voltages2, color='red', label='Solver Voltage', marker='x', alpha=0.6)
    plt.xlabel("Node")
    plt.ylabel("Voltage")
    plt.xticks([])
    plt.title("Voltage Comparison for Each Node")
    plt.legend()
    plt.grid(True)
    plt.savefig("scatter.png", format='png')
    plt.close()

# File paths
file1 = "./benchmarks/real-circuit-data/testcase18/netlist.ic0"  # Replace with the path to your hspice output file
file2 = "./benchmarks/real-circuit-data/testcase18/log.log"  # Replace with the path to your solver output file

data1 = parse_file_1(file1)
data2 = parse_file_2(file2)

scatter_plot(data1, data2)