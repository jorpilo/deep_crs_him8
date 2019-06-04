import numpy as np
from multiprocessing import Pool
import xarray as xr
import os
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import argparse

# kmeans function
# source link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
def Kmeans_one(b7):
    shape = b7.shape
    clusters = 3
    b7 = b7.astype(np.float32)

    flat_image = b7.reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(flat_image,clusters , None, criteria, 10, flags)

    cluster_centers = centers
    cluster_labels = labels
    img = cluster_centers[cluster_labels].reshape(shape)

    return img


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

def process_many(infile, outfile, kmeans):
    dataset = xr.open_dataset(infile)
    datasets = ["B7","B9"]
    ds = xr.Dataset({})
    for number in datasets:
        b7 = dataset[number].values
        if kmeans:
            result = process_batch_kmeans(b7)
        else:
            result = process_batch_meanshift(b7)
        print("Processes done for "+number)

        result = np.array(result, dtype='uint8')
        ds[number] = xr.DataArray(result, dims=['time','width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf(outfile, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical Flow algorithm')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    parser.add_argument("-m", "--mean", help="Performs meanshift instead of kmeans", action="store_true",
                        dest="mean")

    args = parser.parse_args()
    process_many(args.input, args.output, args.mean)
