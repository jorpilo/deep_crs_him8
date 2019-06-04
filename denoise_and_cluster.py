# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:23:13 2019

@author: RMC
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import cluster
from scipy import ndimage

# read in one random image
b7 = np.load("D:/dataset/b9_30.npy")[100]
maxb = np.max(b7)
b7 = b7 / maxb
b7 = b7*255
b7 = b7.astype(np.uint8)

kernel = np.ones((5,5),np.uint8)
# implement fastmeansdenoising 
# source link1: https://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html
# source link2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
# We choose a small number for the size of template winodw here to reserve the details of the clouds
# while implementing the denoising 
def denoise(img):
    dst = cv2.fastNlMeansDenoising(b7,None,3,7)
    return dst

# implememt erosion on denoised image for data enhancement 
# source link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def erosion(img):
    erosion = cv2.erode(img,kernel,iterations = 1)
    return erosion


def sharpen(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Taken from https://github.com/soroushj/python-opencv-numpy-example/blob/master/unsharpmask.py
    Return a sharpened version of the image, using an unsharp mask.
    """
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

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
denoised = denoise(b7)
erosion = erosion(denoised)
sharpened = sharpen(denoised)

plt.figure(figsize = (28,15))
plt.subplot(231),plt.imshow(b7),plt.title("origin")
plt.subplot(232),plt.imshow(denoised),plt.title("denoised")
plt.subplot(233),plt.imshow(erosion),plt.title("erosion")
plt.subplot(234),plt.imshow(sharpened),plt.title("sharpened")

meanshift = Meanshift(b7)
kmean = Kmeans(b7)
plt.figure(figsize = (28,15))
plt.subplot(231),plt.imshow(meanshift),plt.title("Mean shift")
plt.subplot(232),plt.imshow(kmean),plt.title("Kmeans with 5 clusters")