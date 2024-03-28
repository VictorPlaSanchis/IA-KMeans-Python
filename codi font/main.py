import sys
import matplotlib.pyplot as plt
import os
import glob
import os
import time

from kmeans import KMeans
from elbowtest import elbowTest
from data import Data
from validation import validateClusters
from globalVariables import __PLOTS__, __DATASET__, __CENTROIDS_INITIALIZATION__, __CENTROIDS_COMPUTATION__, __ELBOW_TEST_MAX_K__, __GIFS__,__README_DATASET__,__COMPUTE_SILHOUETTE__,__COMPUTE_VALIDATION__

data = Data(__DATASET__,__README_DATASET__)
dataKMeans = data.dataTreatment()
groundtruth = data.getGroundtruth()
k = data.getK()

print("INFORMACION DE EJECUCION:")
print("\t- dataset:", __DATASET__)
print("\t- README dataset:", __README_DATASET__)
print("\t- N =", len(dataKMeans))
print("\t- D =", len(dataKMeans[0]))
print("\t- K =", k)

print("\nINFORMACION DEL README I VARIABLES GLOBALES:")
print("\t- etiquetat =", groundtruth)
print("\t- inicialitzacio de centroides =", __CENTROIDS_INITIALIZATION__)
print("\t- inicialitzacio de centroides =", __CENTROIDS_COMPUTATION__)
print("\t- uso de plots =", __PLOTS__)

if k == None:
    print()
    input("Enter para seguir con el Elbow Test...")
    print()
    print("ELBOW TEST para k = [ 1,", __ELBOW_TEST_MAX_K__, "]:")
    k = elbowTest(__ELBOW_TEST_MAX_K__, dataKMeans)

start = time.time()
algorithm = KMeans(k, dataKMeans, __CENTROIDS_INITIALIZATION__,datasetName=__DATASET__)
its = algorithm.computeKMeans(__CENTROIDS_COMPUTATION__)

print()
print("tiempo de ejecucion =", time.time() - start)
print("iteraciones =", its)

def deleteAllIterations():

    pattern = os.path.join(".", 'iteration*.png')
    pattern2 = os.path.join(".", 'kmeans_animation_*.gif')

    # Obtener la lista de archivos que coinciden con el patrón
    files_to_delete = glob.glob(pattern)
    # Obtener la lista de archivos que coinciden con el patrón
    files_to_delete2 = glob.glob(pattern2)

    # Eliminar los archivos
    for file in files_to_delete:
        os.remove(file)
    for file in files_to_delete2:
        os.remove(file)


if __COMPUTE_SILHOUETTE__:
    print("\nSILHOUETTE:")
    algorithm.printSimilarity()

print()
if __COMPUTE_VALIDATION__:
    print("VALIDACION DE DATOS:")
    if not groundtruth:
        print("No hi ha validacio per aquest dataset en concret.")
    else:
        validateClusters(algorithm.getClusters(), data.getValidation())

if __PLOTS__:
    plt.show()

if __GIFS__:
    input("Dale a ENTER para seguir, se borraran los gifs i pngs generados...")
    deleteAllIterations()