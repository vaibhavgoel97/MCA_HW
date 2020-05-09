import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from os import listdir
from os.path import isfile, join

def getFeatures(img):
    maxValue = np.amax(img) + 1
    distanceSet = [1, 3, 5, 7]
    W = img.shape[0]
    H = img.shape[1]
    histogram = np.zeros(maxValue)
    for i in range(W):
        for j in range(H):
            histogram[img[i][j]] += 1
    distanceSize = len(distanceSet)
    correlogram = np.zeros((maxValue, distanceSize))
    
    for i in range(distanceSize):
        d = distanceSet[i]
        print(d)
        for j in range(W):
            for k in range(H):
                c = img[j][k]
                for dx in range(-d, d + 1):
                    X = j + dx
                    Y = k - d
                    if 0 <= X and X < W and 0 <= Y and Y < H and img[X][Y] == c: 
                        correlogram[c][i]+=1
                        
                    Y = k + d  
                    if 0 <= X and X < W and 0 <= Y and Y < H and img[X][Y] == c:
                        correlogram[c][i]+=1
                        
                    
                for dy in range(-d + 1,d):
                    X = j - d
                    Y = k + dy
                    if 0 <= X and X < W and 0 <= Y and Y < H and img[X][Y] == c: 
                        correlogram[c][i]+=1
                            
                    X = j + d
                    if 0 <= X and X < W and 0 <= Y and Y < H and img[X][Y] == c:
                        correlogram[c][i]+=1
                
        for j in range(maxValue):
            correlogram[j][i] = correlogram[j][i] / (histogram[j] + 1);  
    return correlogram


# #Code for correlogram generation
mypath = "images"
# couter = 0
# features = {}
# for f in listdir(mypath):
#     img = cv2.imread(mypath+"/"+f)
#     print(str(couter) + "::::" + f)  
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#     correlogram = getFeatures(img)
#     features[f] = correlogram
#     print(features)
#     couter+=1

# import pickle
# pickle_features = open("features.pickle","wb")
# pickle.dump(features,pickle_features)

def sortSecond(val): 
    return val[1] 

#Code for evaluation
# mypathquery = "train/query"


import pickle
pickle_features = open("features.pickle","rb")
features = pickle.load(pickle_features)

def getCorr(f):
    img = cv2.imread("images"+"/"+f)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    maxValue = np.amax(img) + 1
    correlogram = getFeatures(img)
    diffDict = {}
    for imagee in features.keys():
        diffDict[imagee] = np.absolute(np.subtract(correlogram[:64,], features[imagee][:64,]))
    addDict = {}
    for imagee in features.keys():
        addDict[imagee] = np.add(1, np.add(correlogram[:64,], features[imagee][:64,]))
    finalDict = []
    for imagee in features.keys():
        val = ((np.divide(diffDict[imagee], addDict[imagee])).sum()/64)
        if not isnan(val):
            finalDict.append([imagee, val])
    finalDict.sort(key = sortSecond)
    return (finalDict[:100])

def findPrecision(query, model):
    c = 0
    for val in model:
        if(val[0][:-4] in query):
            c+=1
    return c*100/len(model)

def findRecall(query, model):
    c = 0
    for val in model:
        if(val[0][:-4] in query):
            c+=1
    return c*100/len(query)
def findF1(precision, recall):
    if not (precision+recall == 0):
        return ((2 * precision * recall)/(precision + recall))
    else:
        return -1

mypath  = "train"
subpath1 = "ground_truth"
subpath = "query"
precisionListGood = []
recallListGood = []
for i in listdir(mypath + "/" + subpath):
    with open(mypath + "/" + subpath + "/" + i, 'rb') as f:
        fname = str(f.read())[7:].split()[0]
    print(fname + ".jpg")
    groupType = "good"

    with open(mypath + "/" + subpath1 + "/" + i[:-9] + groupType + ".txt", 'rb') as f:
        query = str(f.read()).split('\\n')
        query[0] = query[0][2:]
        query = query[:-1]

        model = getCorr(fname + ".jpg")

        precisionListGood.append(findPrecision(query,model))
        recallListGood.append(findRecall(query, model))

precisionListGood.sort()
recallListGood.sort()
print("PRECISION GOOD")
print("Maximum: " + str(precisionListGood[len(precisionListGood) - 1]))
print("Minimum: " + str(precisionListGood[0]))
print("Average: " + str(sum(precisionListGood)/len(precisionListGood)))
print("RECALL GOOD")
print("Maximum: " + str(recallListGood[len(recallListGood) - 1]))
print("Minimum: " + str(recallListGood[0]))
print("Average: " + str(sum(recallListGood)/len(recallListGood)))
print("F1 GOOD")
print("Maximum: " + str(findF1(precisionListGood[len(precisionListGood) - 1], recallListGood[len(recallListGood) - 1])))
print("Minimum: " + str(findF1(precisionListGood[0], recallListGood[0])))
print("Average: " + str(findF1(sum(precisionListGood)/len(precisionListGood), sum(recallListGood)/len(recallListGood))))

precisionListJunk = []
recallListJunk = []
for i in listdir(mypath + "/" + subpath):
    with open(mypath + "/" + subpath + "/" + i, 'rb') as f:
        fname = str(f.read())[7:].split()[0]
    print(fname + ".jpg")
    groupType = "junk"

    with open(mypath + "/" + subpath1 + "/" + i[:-9] + groupType + ".txt", 'rb') as f:
        query = str(f.read()).split('\\n')
        query[0] = query[0][2:]
        query = query[:-1]

        model = getCorr(fname + ".jpg")

        precisionListJunk.append(findPrecision(query,model))
        recallListJunk.append(findRecall(query, model))

precisionListJunk.sort()
recallListJunk.sort()
print("PRECISION JUNK")
print("Maximum: " + str(precisionListJunk[len(precisionListJunk) - 1]))
print("Minimum: " + str(precisionListJunk[0]))
print("Average: " + str(sum(precisionListJunk)/len(precisionListJunk)))
print("RECALL JUNK")
print("Maximum: " + str(recallListJunk[len(recallListJunk) - 1]))
print("Minimum: " + str(recallListJunk[0]))
print("Average: " + str(sum(recallListJunk)/len(recallListJunk)))
print("F1 JUNK")
print("Maximum: " + str(findF1(precisionListJunk[len(precisionListJunk) - 1], recallListJunk[len(recallListJunk) - 1])))
print("Minimum: " + str(findF1(precisionListJunk[0], recallListJunk[0])))
print("Average: " + str(findF1(sum(precisionListJunk)/len(precisionListJunk), sum(recallListJunk)/len(recallListJunk))))

precisionListOK = []
recallListOK = []
for i in listdir(mypath + "/" + subpath):
    with open(mypath + "/" + subpath + "/" + i, 'rb') as f:
        fname = str(f.read())[7:].split()[0]
    print(fname + ".jpg")
    groupType = "ok"

    with open(mypath + "/" + subpath1 + "/" + i[:-9] + groupType + ".txt", 'rb') as f:
        query = str(f.read()).split('\\n')
        query[0] = query[0][2:]
        query = query[:-1]

        model = getCorr(fname + ".jpg")

        precisionListOK.append(findPrecision(query,model))
        recallListOK.append(findRecall(query, model))

precisionListOK.sort()
recallListOK.sort()
print("PRECISION OK")
print("Maximum: " + str(precisionListOK[len(precisionListOK) - 1]))
print("Minimum: " + str(precisionListOK[0]))
print("Average: " + str(sum(precisionListOK)/len(precisionListOK)))
print("RECALL OK")
print("Maximum: " + str(recallListOK[len(recallListOK) - 1]))
print("Minimum: " + str(recallListOK[0]))
print("Average: " + str(sum(recallListOK)/len(recallListOK)))
print("F1 OK")
print("Maximum: " + str(findF1(precisionListOK[len(precisionListOK) - 1], recallListOK[len(recallListOK) - 1])))
print("Minimum: " + str(findF1(precisionListOK[0], recallListOK[0])))
print("Average: " + str(findF1(sum(precisionListOK)/len(precisionListOK), sum(recallListOK)/len(recallListOK))))





