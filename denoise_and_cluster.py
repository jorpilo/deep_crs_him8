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
b7 = np.load("D:/dataset/b9_30.npy")[100]
maxb = np.max(b7)
b7 = b7 / maxb
b7 = b7*255
b7 = b7.astype(np.uint8)

# implement fastmeansdenoising 
# source link1: https://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html
# source link2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
# We choose a small number for the size of template winodw here to reserve the details of the clouds
# while implementing the denoising 

# display the results 
dst = cv2.fastNlMeansDenoising(b7,None,3,7)
plt.figure(figsize = (20,20))
plt.subplot(231),plt.imshow(b7),plt.title("origin")
plt.subplot(232),plt.imshow(dst),plt.title("denoised")

# implememt erosion on denoised image for data enhancement 
# source link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
kernel = np.ones((3,4),np.uint8)
erosion = cv2.erode(dst,kernel,iterations = 1)

#blurring then sharpening 
#source link https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html
f = b7.astype(float)
blurred_f = ndimage.gaussian_filter(f, 3)
filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

# display all the results 
plt.figure(figsize = (28,15))
plt.subplot(231),plt.imshow(b7),plt.title("origin")
plt.subplot(232),plt.imshow(dst),plt.title("denoised")
plt.subplot(233),plt.imshow(erosion),plt.title("erosion")
plt.subplot(234),plt.imshow(filter_blurred_f),plt.title("blurred")
plt.subplot(235),plt.imshow(sharpened),plt.title("sharpened")
plt.show()

# kmeans function 
# source link: https://dzone.com/articles/cluster-image-with-k-means
def Kmeans(img):
    x, y= img.shape
    
    flat_image = img.reshape(-1, 1)
    
    kmeans_cluster = cluster.KMeans(n_clusters=5) # modify cluster number here
    kmeans_cluster.fit(flat_image)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    result = cluster_centers[cluster_labels].reshape(x, y)
    return result

# MeanShift function 
# source link: http://efavdb.com/mean-shift/
def Meanshift(img):
    
    flat_image = img.reshape(-1, 1)
    
    bandwidth2 = estimate_bandwidth(flat_image,
                                    quantile=.2, n_samples=500)
    ms = MeanShift(bandwidth2, bin_seeding=True, cluster_all=False)
    ms.fit(flat_image)
    labels = ms.labels_
    segmented_image = np.reshape(labels, img.shape)
      
    return segmented_image 

#display all the results for comparison     
clo = Kmeans(b7)
cldenoised = Kmeans(dst)
clerosion = Kmeans(erosion)
clblurred = Kmeans(filter_blurred_f)
clsharpened = Kmeans(sharpened)
res = np.load("D:/dataset/crsflux_30.npy")[7]

plt.figure(figsize = (28,15))
plt.subplot(231),plt.imshow(clo),plt.title("origin")
plt.subplot(232),plt.imshow(cldenoised),plt.title("denoised")
plt.subplot(233),plt.imshow(clerosion),plt.title("erosion")
plt.subplot(234),plt.imshow(clblurred),plt.title("blurred")
plt.subplot(235),plt.imshow(clsharpened),plt.title("sharpened")
plt.subplot(236),plt.imshow(res),plt.title("The result we want")


dst = cv2.fastNlMeansDenoising(b7,None,3,7)
dst1 = cv2.fastNlMeansDenoising(b7,None,7,7)
plt.subplot(231),plt.imshow(dst),plt.title("3")
plt.subplot(232),plt.imshow(dst1),plt.title("7")


meanshift = Meanshift(b7)
kmean = Kmeans(b7)
plt.figure(figsize = (28,15))
plt.subplot(231),plt.imshow(meanshift),plt.title("Mean shift")
plt.subplot(232),plt.imshow(kmean),plt.title("Kmeans with 5 clusters")