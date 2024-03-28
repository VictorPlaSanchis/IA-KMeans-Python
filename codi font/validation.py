from randindex import RandIndex

def validateValidation(clusters, validation):
    dic = {}
    for i in range(0,len(validation)):
        if validation[i] not in dic.keys():
            dic[validation[i]] = [i]
        else:
            dic[validation[i]].append(i)

    validationClusters = []
    for key in dic.keys():
        validationClusters.append([])
        for value in dic[key]:
            validationClusters[-1].append(value)

    r = RandIndex(clusters, validationClusters, len(validation))
    print("RAND INDEX:\n",r.compute())

def validateClusters(A, B, N):
    r = RandIndex(A, B, N)
    print("RAND INDEX:\n",r.compute())
