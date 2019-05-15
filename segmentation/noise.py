# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:23:13 2019

@author: RMC
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pylab import imread,subplot,imshow,show
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import normalize
from sklearn import cluster
from scipy import ndimage
img = cv2.imread('b7.png',cv2.IMREAD_ANYDEPTH)

#img = np.load("D:/deep_crs_him8-master/b7_30.npy")[100]


#plt.imshow(b7,cmap='gray')
#plt.imsave("b7.png", b7)
"""
img = cv2.imread('b7.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.imread('b7.png', cv2.IMREAD_GRAYSCALE)
plt.imsave("gray.png", gray)"""

# read in one image
b7 = np.load("D:/dataset/b7_30.npy")[7]
maxb = np.max(b7)
b7 = b7 / maxb
b7 = b7*255
b7 = b7.astype(np.uint8)
# implement fastmeansdenoising 
dst = cv2.fastNlMeansDenoising(b7,None,9,13)

# implememt erosion on denoised image
kernel = np.ones((3,4),np.uint8)
erosion = cv2.erode(dst,kernel,iterations = 1)

#blurred
f = b7.astype(float)
blurred_f = ndimage.gaussian_filter(f, 3)
filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

#sharpened
alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
plt.figure(figsize = (15,8))
plt.subplot(231),plt.imshow(b7),plt.title("origin")
plt.subplot(232),plt.imshow(dst),plt.title("denoised")
plt.subplot(233),plt.imshow(erosion),plt.title("erosion")
plt.subplot(234),plt.imshow(filter_blurred_f),plt.title("blurred")
plt.subplot(235),plt.imshow(sharpened),plt.title("sharpened")
plt.show()


def segment(img):
    x, y= img.shape
    
    flat_image = img.reshape(-1, 1)
    '''
    bandwidth2 = estimate_bandwidth(flat_image,
                                    quantile=.2, n_samples=500)
    print(bandwidth2)
    ms = MeanShift(bandwidth2, bin_seeding=True, cluster_all=False)
    ms.fit(flat_image)
    labels = ms.labels_
    segmented_image = np.reshape(labels, original_shape[:2])
    
    
    plt.imshow(segmented_image)
    '''
    kmeans_cluster = cluster.KMeans(n_clusters=5) # modify cluster number here
    kmeans_cluster.fit(flat_image)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    result = cluster_centers[cluster_labels].reshape(x, y)
    return result
    
clo = segment(b7)
cldenoised = segment(dst)
clerosion = segment(erosion)
clblurred = segment(filter_blurred_f)
clsharpened = segment(sharpened)
res = np.load("D:/dataset/crsflux_30.npy")[7]

plt.figure(figsize = (15,8))
plt.subplot(231),plt.imshow(clo),plt.title("origin")
plt.subplot(232),plt.imshow(cldenoised),plt.title("denoised")
plt.subplot(233),plt.imshow(clerosion),plt.title("erosion")
plt.subplot(234),plt.imshow(clblurred),plt.title("blurred")
plt.subplot(235),plt.imshow(clsharpened),plt.title("sharpened")
plt.subplot(236),plt.imshow(res),plt.title("The result we want")


