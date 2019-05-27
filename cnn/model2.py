from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D
from keras import backend as K
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf

import numpy as np
from utils.IO import *

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 4)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 4)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 4)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 4)))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics=['mae'])
    print(model.summary())
    return model


def main():
    # Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
    x = load_dataset("../dataset/sat_pre_images",[7,9,11,16], normalize=True)[1:, :, :, :]

    # Load measured precipitation to create Y (output of the network)
    y = np.load("../dataset/sat_pre_images/crsflux_30.npy")[1:,:,:,np.newaxis]
    y = np.log(1+y)
    # Verify dimensions of the data.
    # We should have 1163 samples of 400x400 images for both X and Y
    print(x.shape, y.shape)

    # Instantiate model defined in function above
    model = GetModel()
    # Fit data using a 70/30 validation split
    history = model.fit(x, y, epochs=20, verbose=1, validation_split=.20, shuffle=True)

    # Save the model once trained for later use
    model.save('model2.h5')

    # Generate images from the data to see if model makes sense
    y_pred = model.predict(x[:1, :, :, :])
    #plt.imsave("pred_rain_2.png", np.exp(y_pred[0,:,:,0]))
    #plt.imsave("obs_rain_2.png", np.exp(y[0,:,:,0]))
    plt.imsave("pred_rain_2_3_log.png", y_pred[0, :, :, 0])
    plt.imsave("obs_rain_2_3_log.png", y[0,:,:,0])



if __name__ == "__main__":
    main()