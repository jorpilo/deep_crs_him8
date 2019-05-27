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

def Kmeans_one(b7):
    shape = b7.shape
    b7 = b7.astype(np.float32)

    flat_image = b7.reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(flat_image, 6, None, criteria, 10, flags)

    cluster_centers = centers
    cluster_labels = labels
    img = cluster_centers[cluster_labels].reshape(shape)

    return img


def process_many():
    filename = "../dataset/sat_pre_images/kmeans.nc"
    datasets = ["B7","B9"]
    ds = xr.Dataset({})
    for number in datasets:
        path = os.path.join("../dataset/sat_pre_images/", number.lower()+"_30.npy")
        b7 = np.load(path)

        print("Starting threads")
        with Pool(8) as p:
            result = p.map(Kmeans_one, b7)

        print("Processes done for "+number)

        result = np.array(result, dtype='uint8')
        ds[number] = xr.DataArray(result, dims=['time','width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf(filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

if __name__ == "__main__":
    process_many()


