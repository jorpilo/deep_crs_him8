#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   pca.py

   performs pca on a dataset, returns csv files with the
   MSE, euclidian and cosine distances of the datasets

   Usage: pca.py filename [--ncomps]
"""

import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import argparse
import xarray as xr

# For saving/loading already calculated 1-PCA
def save(transformer, avg):
    with open('pca.pickle', 'wb') as handle:
        pickle.dump(transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('avg.pickle', 'wb') as handle:
        pickle.dump(avg, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load():
    with open('pca.pickle', 'rb') as handle:
        transformer = pickle.load(handle)
    with open('avg.pickle', 'rb') as handle:
        avg = pickle.load(handle)
    return transformer, avg




def PCA(dataset, comps):
    ## Calculate the mean
    avg = np.zeros((160000))
    nalltest = 0
    for i in range(7,17):
        data = dataset['B{}'.format(i)].values
        data = data.reshape(data.shape[0],-1)
        avg += np.sum(data, axis=0)
        nalltest += data.shape[0]
    avg = avg / nalltest

    ## Calculate the 1-PCA operator
    tranformer = IncrementalPCA(n_components = 100, batch_size=data.shape[0])
    for i in range(7,17):
        data = dataset['B{}'.format(i)].values
        data = data.reshape(data.shape[0],-1)
        data = data - avg
        tranformer.partial_fit(data)
        print("loaded {}".format(i))

    ## Start comparing matrix

    ## MSE
    distance_matrix = np.zeros((10,10))
    for i in range(7,17):
        for j in range(i+1,17):
            if i != j:
                data = dataset['B{}'.format(i)].values
                data = data.reshape(data.shape[0],-1)
                data1 = tranformer.transform(data-avg)
                data = dataset['B{}'.format(j)].values
                data = data.reshape(data.shape[0],-1)
                data2 = tranformer.transform(data-avg)
                distance = 0
                for z in range(data.shape[0]):
                    distance += mean_squared_error(data1[z], data2[z])
                distance_matrix[i-7,j-7] = distance
                print(str(i)+"-> "+str(j)+" = "+str(distance))

    ## Euclidian distance
    distance_matrix2 = np.zeros((10,10))
    for i in range(7,17):
        for j in range(i+1,17):
            data = dataset['B{}'.format(i)].values
            data = data.reshape(data.shape[0],-1)
            data1 = tranformer.transform(data-avg)
            data = dataset['B{}'.format(j)].values
            data = data.reshape(data.shape[0],-1)
            data2 = tranformer.transform(data-avg)
            distance = 0
            for z in range(data.shape[0]):
                distance += euclidean(data1[z], data2[z])
            distance_matrix2[i-7,j-7] = distance
            print(str(i)+"-> "+str(j)+" = "+str(distance))

    ## Cosine distance
    distance_matrix3 = np.zeros((10,10))
    for i in range(7,17):
        for j in range(i+1,17):
            if i != j:
                data = dataset['B{}'.format(i)].values
                data = data.reshape(data.shape[0],-1)
                data1 = tranformer.transform(data-avg)
                data = dataset['B{}'.format(j)].values
                data = data.reshape(data.shape[0],-1)
                data2 = tranformer.transform(data-avg)
                distance = 0
                for z in range(data.shape[0]):
                    distance += cosine(data1[z], data2[z])
                distance_matrix3[i-7,j-7] = distance
                print(str(i)+"-> "+str(j)+" = "+str(distance))
    np.savetxt("mse.csv", distance_matrix, delimiter=",")
    np.savetxt("euclidian.csv", distance_matrix2, delimiter=",")
    np.savetxt("cosine.csv", distance_matrix3, delimiter=",")

## Main
if __name__ == "__main__":
    # Arguments:
    parser = argparse.ArgumentParser(description='Calculates the PCA components of a dataset')
    parser.add_argument('filename',
                        help='Name to the dataset to calculate the PCA on', type=str)
    parser.add_argument('--ncomp',help='number of components to calculate, default 100', type=int, default=100, dest="ncomp")
    args = parser.parse_args()

    # Load dataset
    dataset = xr.open_dataset(args.filename)

    # Perform 1-PCA
    PCA(dataset, args.ncomp)


