# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:43:02 2019

@author: RMC
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage
from pylab import imread,subplot,imshow,show
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import normalize
from sklearn import cluster
from multiprocessing import Pool

def process_one(b7):
    x, y= b7.shape
    b7 = b7.astype(np.uint8)
    f = b7.astype(float)
    blurred_f = ndimage.gaussian_filter(f, 3)

    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

    alpha = 30
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    '''
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(f)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(blurred_f)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(sharpened)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    '''
    flat_image = sharpened.reshape(-1, 1)
    kmeans_cluster = cluster.KMeans(n_clusters=6)
    kmeans_cluster.fit(flat_image)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    img = cluster_centers[cluster_labels].reshape(x, y)
    '''
    plt.figure(figsize = (15,8))
    plt.imshow(img)
    '''
    print("hey")
    return img



filename = "newb10_30"    
b7 = np.load("D:/dataset/b10_30.npy")

maxb = np.amax(b7)
minb = np.amin(b7)
b7 = (b7-minb) / (maxb-minb)
b7 = b7*255

print("Starting threads")
p = Pool(2)
result = p.map(process_one, b7[0:5])
print("Processes done")
data = np.array(result, dtype='uint8')
    #Convert the new npy file to png
    
np.save(filename + '.npy', data)
print(filename + " was saved")
'''
res = np.load("D:/dataset/crsflux_30.npy")[7]
plt.figure(figsize = (15,8))
plt.imshow(res)
'''
