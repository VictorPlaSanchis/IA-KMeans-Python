import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(K, N, D):
    data = []
    for _ in range(K):
        # Generar centroides aleatorios
        clusters = []
        for _ in range(D):
            centroid = np.random.uniform(-10, 10)
            # Generar datos alrededor de los centroides
            clusters.append(np.random.normal(centroid, 1, N // K))
        # Agregar los datos al conjunto de datos
        data.extend(list(zip(*clusters)))
    return np.array(data)

if len(sys.argv) != 4:
    print("Usage: python script.py <K> <N> <D>")
    sys.exit(1)

K = int(sys.argv[1])
N = int(sys.argv[2])
D = int(sys.argv[3])

columns = [str(i) for i in range(D)]
data = generate_data(K, N, D)
df = pd.DataFrame(data)
filename = f"dataset-{K}-{N}-{D}.csv"
df.to_csv(filename, index=False, header=False, sep=';')

with open(f"README-{filename[:-4]}.txt", "w") as readme_file:
    readme_file.write(f"Nombre d'instàncies: n = {len(df)}\n")
    readme_file.write(f"Dimensions ('features'): d = {D}\n")
    readme_file.write(f"Nombre de clústers: k = {K}\n")
    readme_file.write("Etiquetat: Sí\n")

if D == 2:
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=10)
    plt.title(f'Conjunto de datos con {K} clusters bien diferenciados')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

print(f"Dataset guardado en '{filename}'")
print(f"README generado como 'README-{filename[:-4]}.txt'")
