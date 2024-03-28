import csv
import numpy as np
import sys
from sklearn.decomposition import PCA

class Data:

  data = []
  treated = []
  K = None
  groundtruth = None
  validation = []

  def __init__(self,filename,readme):
    with open(filename, newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=';')
      for row in reader:
          values = []
          for value in row:
            values.append(float(value.replace(',','.')))
          self.data.append(values)

    with open(readme, 'r') as archivo:
      lineas = archivo.readlines()
      for linea in lineas:
        if "k" in linea:
            valorK = linea.split('=')[1].strip().split()[0]
            if valorK == "??":
              self.K = None
            else:
              self.K = int(valorK)
        elif "Etiquetat" in linea:
            etiqueta = linea.split(':')[1].strip()
            self.groundtruth = etiqueta.lower()[0] == "s"

  def transpose(self, data):
    return [list(columna) for columna in zip(*data)]

  def normalize(self):
    # escalado min-max
    for i in range(0,len(self.treated)):
      maxV = -sys.maxsize
      minV = sys.maxsize
      for point in self.treated[i]:
        if point > maxV:
          maxV = point
        elif point < minV:
          minV = point
      for i in range(0,len(self.treated)):
        for j in range(0,len(self.treated[i])):
          self.treated[i][j] = ((self.treated[i][j] - minV) / (maxV - minV))

  def removeAtipicValues(self):

    ranges = []

    for i in range(0,len(self.treated)):
      sortedData = sorted(self.treated[i])
      n = len(self.treated[i]) + 1
      Q1 = None
      if int(n/4) != float(n)/4:
        Q1 = (sortedData[int(n/4)] + sortedData[int(n/4)+1]) / 2.0
      else:
        Q1 = sortedData[int(n/4)]

      Q3 = None
      if (3*int(n/4)) != (3*float(n)/4):
        Q3 = (sortedData[3*int(n/4)] + sortedData[3*int(n/4)+1]) / 2.0
      else:
        Q3 = sortedData[3*int(n/4)]

      IQR = Q1 - Q3
      ranges.append((
        Q1 + 1.5 * IQR,
        Q3 - 1.5 * IQR
      ))

    atipics = {}
    for i in range(0,len(self.treated)):
      for j in range(0,len(self.treated[i])):
        if j not in atipics:
            atipics[j] = []
        if self.treated[i][j] < ranges[i][0]:
          atipics[j].append(i)
        elif self.treated[i][j] > ranges[i][1]:
          atipics[j].append(i)
    
    indexsToRemove = []
    newData = []
    for element in atipics.keys():
      if len(atipics[element]) > (len(self.treated)/2):
        indexsToRemove.append(element)

    newData = []
    for i in range(len(self.treated)):
        newRow = []
        if isinstance(self.treated[i], list):
            for j in range(len(self.treated[i])):
                if j not in indexsToRemove:
                    newRow.append(self.treated[i][j])
            newData.append(newRow)

    self.treated = newData

  def PCA(self,dimensions):
    # transpose
    self.treated = self.transpose(self.treated)
    pca = PCA(n_components=dimensions) 
    self.treated = pca.fit_transform(np.array(self.treated))
    # transpose
    self.treated = self.transpose(self.treated)

  def dataTreatment(self):

    # transpose
    self.data = self.transpose(self.data)

    if self.groundtruth:
      self.treated = self.data[0:-1]
      self.validation = self.data[-1]
    else:
      self.treated = self.data

    # normalize data
    self.normalize()

    # PCA
    #dim = 2
    #self.PCA(dim)

    # transpose
    self.treated = self.transpose(self.treated)
    self.data = self.transpose(self.data)
    return self.treated

  def getK(self):
    return self.K

  def getGroundtruth(self):
    return self.groundtruth

  def getValidation(self):
    return self.validation
