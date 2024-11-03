import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = './timing.xlsx'
data = pd.read_excel(file_path)

data['Benchmark (Number of Nodes)'] = data['benchmark'] + ' (' + data['Number of nodes'].astype(str) + ')'

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))

time_columns = [
    'Solver Parser Time (sec)', 
    'Solver Matrices Builder Time (sec)', 
    'Solver Solving Time (sec)', 
    'Solver Results Printing Time (sec)', 
    'Solver Total Time (sec)'
]

for column in time_columns:
    plt.plot(data['Benchmark (Number of Nodes)'], data[column], marker='o', label=column)

plt.xlabel('Benchmark (Number of Nodes)')
plt.ylabel('Time (sec)')
plt.title('Solver Times vs. Benchmark with Number of Nodes')
plt.xticks(rotation=45, ha='right') 
plt.legend()
plt.grid(True)

plt.tight_layout() 
plt.savefig("timing.png", format='png')
