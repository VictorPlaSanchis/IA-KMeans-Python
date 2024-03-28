import os
import re
import sys
from collections import defaultdict

def extract_k(filename):
    match = re.search(r'K(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

execution_data = defaultdict(list)

if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)
directory_path = sys.argv[1]

for filename in os.listdir(directory_path):
    if filename.startswith('output-'):
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            time = None
            iterations = None
            for line in lines:
                if line.startswith('tiempo de ejecucion ='):
                    match = re.search(r'\d+\.\d+', line)
                    if match:
                        time = float(match.group())
                elif line.startswith('iteraciones ='):
                    match = re.search(r'\d+', line)
                    if match:
                        iterations = int(match.group())
            k = extract_k(filename)
            if k is not None:
                if time is not None and iterations is not None:
                    execution_data[k].append((time, iterations))
                else:
                    print(f"Warning: Missing data for file {filename}")

with open("medias.out", "w") as outfile:
    outfile.write("K\tMedia Tiempo de Ejecucion\tMedia Iteraciones\n")
    for k in sorted(execution_data.keys()):  # Ordenar los valores de K
        data = execution_data[k]
        if data:
            times, iterations = zip(*data)
            average_time = sum(times) / len(times)
            average_iterations = sum(iterations) / len(iterations)
            outfile.write(f"{k}\t{average_time:.6f}\t{average_iterations:.6f}\n")
        else:
            print(f"No hay datos suficientes para calcular medias para K={k}.")
