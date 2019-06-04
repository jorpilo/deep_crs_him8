"""
    utils.py

    helpers for the frame prediction network, dataGenerators and Batch Generators
"""

import numpy as np
from keras.utils import Sequence
from opticalflow import warp_image_all_layers

def dataGenerator(n, learning_frames=2, steps=1):
    training = []
    testing = []

    last = 0
    for i in range(learning_frames,n, learning_frames):
        if i+steps < n:
            training.append(np.arange(last,i))
            testing.append(i+steps)
            last = i
    return np.asarray(training), np.asarray(testing)


class Generator_Flow(Sequence):

    def __init__(self, x_set, y_set, data1, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.data1 = data1
        self.data2 = self.__calculate_flow()

    def __calculate_flow(self):
        of = []
        for i in range(len(self)):
            X = self.__get_x(i)
            for element in X:
                newimage = warp_image_all_layers(element[-2], element[-1])
                of.append(newimage)
        return of

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __get_x(self,idx):
        batch_x = self.data1[self.x[idx * self.batch_size:(idx + 1) * self.batch_size]]
        return batch_x

    def __getitem__(self, idx):
        batch_x = self.data1[self.x[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = self.data1[self.y[idx * self.batch_size:(idx + 1) * self.batch_size]]

        newimages = np.asarray(self.data2[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_x = np.concatenate((batch_x, newimages[:,np.newaxis, :, :]), axis=1)
        return batch_x, batch_y