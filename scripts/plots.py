import matplotlib.pyplot as plt

data = []
dataI = []
with open("medias.out", "r") as file:
    next(file)  # Saltar la primera línea (encabezado)
    for line in file:
        k, time, iteraciones = line.split()
        data.append((int(k), float(time)))
        dataI.append((float(iteraciones),float(time)))

data.sort()
dataI.sort()

ks, times = zip(*data)
its, times = zip(*dataI)

plt.figure(figsize=(10, 6))
plt.bar(ks, times, color='blue')
plt.xlabel('X')
plt.ylabel("Y")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(ks,its, marker='o', color='blue', markersize=6, linestyle='-')
plt.xlabel("X")
plt.ylabel('Y')
plt.grid(True)
plt.tight_layout()
plt.show()