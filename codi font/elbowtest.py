import numpy as np
import time
import matplotlib.pyplot as plt

from globalVariables import __KMEANS__, __CENTROIDS_COMPUTATION__, __PLOTS__, __PLOTS_ELBOW_TEST__

if not __KMEANS__:
  from kmeans import KMeans
else:
  __KMEANS__ = True

def elbowTest(KMAX,dataset):
  global __PLOTS__
  __PLOTS__ = False
  inertias = []
  for k in range(1,KMAX+1):
    start = time.time()
    algorithm = KMeans(k, dataset, "++",isKmeans=True)
    iterations = algorithm.computeKMeans(__CENTROIDS_COMPUTATION__,__PLOTS_ELBOW_TEST__)
    inertias.append(algorithm.intraClusterDistance())
    tiempo = time.time()-start
    print(f"k = {k} en {tiempo:.2f} segundos i {iterations} iteraciones.")

  # Calcular las diferencias entre las inercias
  differences = np.abs(np.diff(inertias))
  relations = []
  for i in range(0,len(differences)-1):
    relations.append(2 * differences[i] * differences[i]/differences[i+1])
  
  # Encontrar el punto donde la tasa de cambio de la inercia comienza a disminuir
  optimal_k = np.argmax(relations) + 2

  print("Elbow test recomienda usar k =",optimal_k)

  __PLOTS__ = True
  plt.plot(range(1,len(inertias)+1), inertias, 'bx-')
  plt.xlabel('K')
  plt.ylabel('intra-cluster distance')
  plt.show()

  return optimal_k

