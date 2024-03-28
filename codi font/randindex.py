
class RandIndex:

  diccionario_A = {}
  diccionario_B = {}
  A = None
  B = None
  dataSize = None

  def __init__(self, A, B, dataSize):
    self.A = A
    self.B = B
    self.dataSize = dataSize
    self.diccionario_A = {}
    self.diccionario_B = {}
    for indice, cluster in enumerate(A):
        for elemento in cluster:
            self.diccionario_A[elemento] = indice
    for indice, cluster in enumerate(B):
        for elemento in cluster:
            self.diccionario_B[elemento] = indice
    return
  
  def compute(self):
    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(0,self.dataSize):
      for j in range(i+1,self.dataSize):
        ai = self.diccionario_A.get(i, -1)
        aj = self.diccionario_A.get(j, -1)
        bi = self.diccionario_B.get(i, -1)
        bj = self.diccionario_B.get(j, -1)

        if ai == aj:
          if bi == bj:
            a += 1
          else:
            c += 1
        else:
          if bi == bj:
            d += 1
          else:
            b += 1

    return float(a+b) / float(a+b+c+d)
