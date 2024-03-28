from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import time
import sys

from globalVariables import __PLOTS__, __KMEANS__, __GIFS__, __DIMENSION_TO_VISUALIZE__

__KMEANS__ = True

# euclidian D-dimensional distance formula
# we assume that dimension(x) == dimension(y), unless the function gets an error
def distance(x,y):
  distance = 0
  for i in range(0,len(x)):
    distance += pow(x[i]-y[i],2)
  return math.sqrt(distance)

class KMeans:

  def __init__(self):
    raise ValueError("DONT USE THIS CONSTRUCTOR")
    return

  def getClusters(self):
    return self.clusters
  
  def getDataSize(self):
    return len(self.data)

  # class constructor
  # reads data from filename
  def __init__(self, K, data, init="Random",datasetName="None",isKmeans=False):
    self.data = []
    self.datasetName = datasetName
    self.D = None
    self.clusters = []
    self.neighbourClusters = {}
    self.old_clusters = []
    self.centroids = []
    self.distances = []
    self.silhouette = None
    self.inertia = None
    self.K = K
    self.centroids = []
    self.initialization = init
    self.isKmeans = isKmeans
    for i in range(0, len(self.data)):
      distance.append([])
      for j in range(0, self.K):
        distance[i].append(None)
    for i in range(0,self.K):
      self.clusters.append([])

    self.data = data
    self.D = len(data[0])

    self.silhouette = Silhouette(self.data)
    return

  # normalize all data points to [0,1] range values
  def normalizeData(self):
    #transpose data
    data = [list(columna) for columna in zip(*self.data)]
    for i in range(0,len(data)):
      maxValue = max(data[i])
      minValue = min(data[i])
      for j in range(len(data[i])):
        # save the data without transposed [j,i] instead of [i,j]
        self.data[j][i] = (data[i][j]-minValue)/(maxValue-minValue)
    #transpose data
    data = [list(columna) for columna in zip(*self.data)]

  # init centroids
  # we use K random data points as initial centroids
  def initCentroids(self):
    # Random centroids
    for _ in range(0,self.K):
      try:
        randInt = random.randint(0, len(self.data)-1)
        self.centroids.append(self.data[randInt])
      except:
        print(randInt)
        self.centroids.append(self.data[random.randint(0, len(self.data)-1)])
         
  # init centroids
  # we use K random data points as initial centroids
  def initCentroidsPlusPlus(self):
    # Selecciona el primer centroide aleatoriamente
    self.centroids = [random.choice(self.data)]

    # Selecciona los restantes k-1 centroides
    for _ in range(1, self.K):
        distances = [min(distance(c, x)**2 for c in self.centroids) for x in self.data]
        total_distance = sum(distances)**2
        probabilities = [d / total_distance for d in distances]
        
        # Selecciona un nuevo centroide basado en las probabilidades calculadas
        new_centroid_index = random.choices(range(len(self.data)), weights=probabilities)[0]
        self.centroids.append(self.data[new_centroid_index])

  # compute of clusters
  # we use euclidian distance to compute the closest centroid of every data point
  def computeClusters(self):
    self.old_clusters = list(self.clusters) # K
    self.clusters = []
    self.neighbourClusters = {}
    for i in range(0,self.K): # K
      self.clusters.append([]) # 1
    for i in range(0,len(self.data)): # n
      # compute distances with every centroids
      lowestDistance = sys.maxsize
      second_distance = sys.maxsize
      closerCluster = 0 
      neighbourCluster = 0
      for j in range(0,self.K): # K
        current_distance = distance(self.data[i],self.centroids[j]) # d
        #self.distances[i][j] = current_distance
        if lowestDistance > current_distance:
          lowestDistance = current_distance
          neighbourCluster = closerCluster
          closerCluster = j
        elif second_distance > current_distance:
          second_distance = current_distance
          neighbourCluster = j
           

      # add point to the closest centroid (if centroid is 1, then added in cluster 1, there is a centroid for every cluster)
      self.clusters[closerCluster].append(i)
      if self.K == 1:
        self.neighbourClusters[i] = closerCluster
      else:
        self.neighbourClusters[i] = neighbourCluster

  # weight of a point respect his centroid for computeCentroidsWeights
  def weight(self, centroid, point):
    if centroid == point:
      return 0.0
    return 1.0 / pow(distance(centroid, point),2)

  # compute of centroids
  # K-Means (the new centroid it will be the mean of every cluster multiplied for some weigth)
  def computeCentroidsWeights(self):
    for i in range(0,self.K):
      sumWeights = 0.0
      new_centroid = [0] * len(self.centroids[i])
      for point in self.clusters[i]:
        weight = self.weight(self.centroids[i],self.data[point])
        sumWeights += weight
        for j in range(0,len(self.data[point])):
          new_centroid[j] += self.data[point][j] * weight
      for j in range(0, len(new_centroid)):
        new_centroid[j] /= sumWeights
      self.centroids[i] = new_centroid

  # compute of centroids
  # K-Means (the new centroid it will be the mean of every cluster)
  def computeCentroids(self):
    for i in range(0,self.K):
      new_centroid = [0] * self.D
      for point in self.clusters[i]: # N
        for j in range(0,self.D): # D
          new_centroid[j] += self.data[point][j]
      for j in range(0, self.D):
        if len(self.clusters[i]) == 0:
           continue
        new_centroid[j] /= len(self.clusters[i])
      self.centroids[i] = new_centroid

  def stopCondition(self):
      if len(self.clusters) != len(self.old_clusters):
          return False

      for i in range(len(self.clusters)):
          if set(self.clusters[i]) != set(self.old_clusters[i]):
              return False

      return True

  def showClusteredPlot2D(self, rowsToShow, iteration):

      # Define colormap
      colormap = plt.cm.viridis

      for cluster_id, cluster in enumerate(self.clusters):
          cluster_data = [[], [], []]
          for i in cluster:
              cluster_data[0].append(self.data[i][rowsToShow[0]])
              cluster_data[1].append(self.data[i][rowsToShow[1]])
              #cluster_data[2].append(self.data[i][rowsToShow[2]])
          if not cluster_data:
              print(f"Warning: Insufficient data points in Cluster {cluster_id + 1}")
              continue
          
          color = colormap(cluster_id / len(self.clusters))    

          # Plot cluster data
          plt.scatter(cluster_data[0], cluster_data[1], c=[color], s=12, alpha=1, label='Cluster', zorder=-sys.maxsize)

      # Plot centroid after all clusters are plotted
      for i in range(0,len(self.centroids)):
          plt.scatter(self.centroids[i][0], self.centroids[i][1], c='red', marker='o', s=50, alpha=0.8, label='Centroides', zorder=sys.maxsize)

      #ax.legend()
      plt.title("Iteration "+str(iteration))
      # Save the plot as an image
      plt.savefig(f'iteration_{iteration}_{self.K}.png')
      plt.close()

  def showClusteredPlot3D(self, rowsToShow, iteration):

      # Define colormap
      colormap = plt.cm.viridis

      for cluster_id, cluster in enumerate(self.clusters):
          cluster_data = [[], [], []]
          for i in cluster:
              cluster_data[0].append(self.data[i][rowsToShow[0]])
              cluster_data[1].append(self.data[i][rowsToShow[1]])
              cluster_data[2].append(self.data[i][rowsToShow[2]])
          if not cluster_data:
              print(f"Warning: Insufficient data points in Cluster {cluster_id + 1}")
              continue
          
          color = colormap(cluster_id / len(self.clusters))    
          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          # Plot cluster data
          ax.scatter(cluster_data[0], cluster_data[1], c=[color], s=12, alpha=1, label='Cluster', zorder=-sys.maxsize)

      # Plot centroid after all clusters are plotted
      for i in range(0,len(self.centroids)):
          ax.scatter(self.centroids[i][rowsToShow[0]], self.centroids[i][rowsToShow[1]], self.centroids[i][rowsToShow[2]], c='red', marker='o', s=50, alpha=0.8, label='Centroides', zorder=sys.maxsize)

      #ax.legend()
      plt.title("Iteration "+str(iteration))
      # Save the plot as an image
      plt.savefig(f'iteration_{iteration}_{self.K}.png')
      plt.close()

  def showClusteredPlot(self, vis, it):
     if len(self.centroids[0]) == 2:
        return self.showClusteredPlot2D(vis,it)
     else:
        return self.showClusteredPlot2D(vis,it)

  def computeKMeans(self, centroidComputation, plots=__PLOTS__):
      visualizeDimensions = __DIMENSION_TO_VISUALIZE__

      # Initialize centroids
      if self.initialization == "Random":
          self.initCentroids()
      elif self.initialization == "++":
          self.initCentroidsPlusPlus()
      else:
          raise ValueError('Variable global __CENTROIDS_INITIALIZATION__ mal setteada!')

      # Compute first clusters
      self.computeClusters()
      # Generate frames for the animation
      iterations = 0
      while not self.stopCondition():
          # Show clustered plot and save as image
          if plots:
              self.showClusteredPlot(visualizeDimensions, iterations)
          # Compute centroids
          if centroidComputation == 'Intra-Cluster Mean':
              self.computeCentroids()
          elif centroidComputation == 'Intra-Cluster Weight Mean':
              self.computeCentroidsWeights()
          else:
              raise ValueError('Variable global __CENTROIDS_COMPUTATION__ mal setteada!')

          # Compute clusters
          self.computeClusters()
          iterations += 1

      if __GIFS__ and not self.isKmeans:
        # Generate GIF from saved frames
        images = []
        for i in range(0, iterations):
            filename = f'iteration_{i}_{self.K}.png'
            images.append(Image.open(filename))
        images[0].save(f'kmeans_animation_{self.K}.gif', save_all=True, append_images=images[1:], optimize=False, duration=3000.0/iterations , loop=0)

      return iterations-1

  def printSimilarity(self):
      print("Computing similarity...")
      values = []
      for cluster in self.clusters:
          cluster_values = []
          for point in cluster:
              neigh = self.neighbourClusters[point]
              cluster_values.append(self.silhouette.silhouetteValue(point, cluster, self.clusters[neigh]))
          values.append(cluster_values)
          print("Cluster computed")

      colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

      width = 0.25
      for color, cluster in enumerate(values):
          # Create a new figure
          fig = plt.figure()
          ax = fig.add_subplot(111)
          # Define colormap
          ax.scatter(width * color + np.arange(len(cluster)), cluster, width, color=colors[color], label="Cluster " + str(color + 1))
          plt.legend()
          
          # Save plot as PNG with dataset name
          dataset_name = self.datasetName
          filename = f"{dataset_name}-cluster{color}-quality.png"
          plt.savefig("./plotsQuality/"+filename)
          plt.close()  # Close the plot to avoid displaying it
          
          print(f"Plot saved as {filename}")


  def getInertia(self):
    if self.inertia == None:
      return self.computeInertia()
    else:
      return self.inertia

  def computeInertia(self):
    start = time.time()
    self.inertia = 0
    for i in range(0, len(self.clusters)):
      for point in self.clusters[i]:
        self.inertia += math.pow(distance(self.data[point],self.centroids[i]),2)
    print("Inertia: ",time.time()-start)
    return self.inertia

  def intraClusterDistance(self):
      total_distance = 0
      
      for i in range(len(self.clusters)):
          centroid = self.centroids[i]
          cluster_points = self.clusters[i]
          
          for point in cluster_points:
              total_distance += math.sqrt(sum((self.data[point][d] - centroid[d]) ** 2 for d in range(len(self.data[0]))))
      
      avg_intra_cluster_distance = total_distance / sum(len(cluster_points) for cluster_points in self.clusters) if self.clusters else 0
      
      return avg_intra_cluster_distance

  def getK(self):
     return self.K

  def getCentroids(self):
     return self.centroids

class Silhouette:

  data = []
  matrixDistances = []

  def __init__(self, data):
    self.data = data
    return
    
  def computeMatrixDistances(self):
    self.matrixDistances = [[0] * len(self.data) for _ in range(len(self.data))]
    num = 0
    for i in range(0, len(self.data)):
        num += 1
        if num % 1000 == 0:
          print(f"{100 * num / len(self.data):.2f}%")
        for j in range(i + 1, len(self.data)):
            distance_val = distance(self.data[i], self.data[j])
            self.matrixDistances[i][j] = distance_val
            self.matrixDistances[j][i] = distance_val  # Matriu triangular, per evitar massa us de memoria
    print("Matrix computed")

  def silhouetteValue(self, value, cluster, neighbourCluster):

    if len(cluster) == 1:
      return 0.0

    a = 0.0
    for point in cluster:
      if value != point:
        i = min(value,point)
        j = max(value,point)
        a += distance(self.data[i],self.data[j])
    a = float(a) / (len(cluster) - 1)

    b = 0.0
    for point in neighbourCluster:
      i = min(value,point)
      j = max(value,point)
      b += distance(self.data[i],self.data[j])
    b = float(b) / len(neighbourCluster)

    return (b - a) / max(a,b)
