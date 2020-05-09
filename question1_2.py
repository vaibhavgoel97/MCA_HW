import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import json
from os import listdir

k = 1.259

def findLoG(img):
    logImages = [] 
    for i in range(0,5):
        sigma = np.power(k,i)
        image = ndimage.gaussian_laplace(img, sigma = sigma)    
        image = np.square(image) 
        logImages.append(image)
    return np.array(logImages)

def detectBlob(LoGImage):
    co_ordinates = [] 
    h = img.shape[0]
    w = img.shape[1]
    for i in range(1,h):
        for j in range(1,w):
            windowImage = LoGImage[:,i-1:i+2,j-1:j+2]
            result = np.amax(windowImage) 
            if result >= 0.03: 
                index, x, y = np.unravel_index(np.argmax(windowImage, axis = None),windowImage.shape)
                co_ordinates.append((int(i+x-1),int(j+y-1)))
    return co_ordinates


mypath = "images"
features = []
data = {}
couter = 0
for f in listdir(mypath):
    img = cv2.imread(mypath+"/"+f)
    print(str(couter) + "::::" + f) 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img/255.0  
    LoGImage = findLoG(img)
    co_ordinates = list(set(detectBlob(LoGImage)))
    data[f] = co_ordinates
    print(data)
    couter+=1

with open('logFeatures.txt', 'w') as outfile:
    json.dump(data, outfile)
