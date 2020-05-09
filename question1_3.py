import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_det
import math
import json
from os import listdir

k = 1.259

def findHessian(img):
    hessianImages = [] 
    for i in range(0,5):
        sigma = np.power(k,i)
        hessianDet = hessian_matrix_det(img, sigma=sigma)
        hessianImages.append(hessianDet)
    return np.array(hessianImages)

def detectKeyPoints(hessianImage):
    co_ordinates = [] 
    h = img.shape[0]
    w = img.shape[1]
    for i in range(2,h):
        for j in range(2,w):
            windowImage = hessianImage[:,i-2:i+2,j-2:j+2]
            result = np.amax(windowImage) 
            if result >= 0.003: 
                index, x, y = np.unravel_index(np.argmax(windowImage, axis = None),windowImage.shape)
                co_ordinates.append((int(i+x-1),int(j+y-1)))
    return co_ordinates


mypath = "images"
features = []
data = {}
couter = 0
# f = "all_souls_000188.jpg"#isko hata dena
for f in listdir(mypath):
    img = cv2.imread(mypath+"/"+f)
    print(str(couter) + "::::" + f) 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    hessianImage = findHessian(img)
    co_ordinates = list(set(detectKeyPoints(hessianImage)))
    data[f] = co_ordinates
    print(data)
    couter+=1

with open('logFeaturesSURF.txt', 'w') as outfile:
    json.dump(data, outfile)
