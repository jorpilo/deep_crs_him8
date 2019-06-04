# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:43:02 2019

@author: RMC
"""

import numpy as np
from multiprocessing import Pool
import xarray as xr
import os
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import argparse

# MeanShift function 
# source link: http://efavdb.com/mean-shift/
def MeanShift_one(b7):

    shape = b7.shape
    flat_image = b7.reshape(-1, 1)
    bandwidth2 = estimate_bandwidth(flat_image,
                                    quantile=.2, n_samples=500)
    ms = MeanShift(bandwidth2, bin_seeding=True, cluster_all=False)
    ms.fit(flat_image)
    labels = ms.labels_
    segmented_image = np.reshape(labels, shape)

    return segmented_image

# kmeans function 
# source link: https://dzone.com/articles/cluster-image-with-k-means
def Kmeans_one(b7):
    shape = b7.shape
    clusters = 3 # modify the cluster number here 
    b7 = b7.astype(np.float32)

    flat_image = b7.reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(flat_image,clusters , None, criteria, 10, flags)

    cluster_centers = centers
    cluster_labels = labels
    img = cluster_centers[cluster_labels].reshape(shape)

    return img

def process_batch_meanshift(batch):

    with Pool(8) as p:
        result = p.map(MeanShift_one, batch)
    result = np.asarray(result)

    return result

def process_batch_kmeans(batch):

    with Pool(8) as p:
        result = p.map(Kmeans_one, batch)
    result = np.asarray(result)

    return result

def kmeans_many(input_filename):
    output_filename = "../dataset/sat_pre_images/kmeans.nc"
    ds = xr.open_dataset(input_filename)
    datasets = ["B7","B9"]
    dx = xr.Dataset({})
    for number in datasets:
        a = ds[number]
        print("Starting threads")
        with Pool(8) as p:
            result = p.map(Kmeans_one, a[0:20])

        print("Processes done for "+number)

        result = np.array(result, dtype='uint8')
        dx[number] = xr.DataArray(result, dims=['time','width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dx.data_vars}

    dx.to_netcdf(output_filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)
    
def meanshift_many(input_filename):
    output_filename = "../dataset/sat_pre_images/meanshift.nc"
    ds = xr.open_dataset(input_filename)
    datasets = ["B7","B9"]
    dx = xr.Dataset({})
    for number in datasets:
        a = ds[number]
        print("Starting threads")
        with Pool(8) as p:
            result = p.map(MeanShift_one, a[0:20])

        print("Processes done for "+number)

        result = np.array(result, dtype='uint8')
        dx[number] = xr.DataArray(result, dims=['time','width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dx.data_vars}

    dx.to_netcdf(output_filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

if __name__ == "__main__":
    # Arguments:
    parser = argparse.ArgumentParser(description='Implement clustering on a dataset')
    parser.add_argument('filename',
                        help='Name to the dataset to implenment clustering on', type=str, metavar="filename")
    args = parser.parse_args()

    # Load dataset
    dataset = xr.open_dataset(args.filename)

    # Perform kmeans and meanshift
    kmeans_many(dataset)
    meanshift_many(dataset)




