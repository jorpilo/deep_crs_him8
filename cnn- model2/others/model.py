from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf

import numpy as np
from utils.IO import *

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, =(400, 400, 4)))
    # Size 400x400x3
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 200x200x32
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 100x100x64
    model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 50x50x128
    model.add(Conv2D(256, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 25x25x256
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 50x50x128
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 100x100x64
    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 200x200x32
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    # Size 400x400x1)
    model.add(Conv2DTranspose(1, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same'))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics=['mae'])

    print(model.summary())

    return model

def main():
    # Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
    x = load_dataset("../dataset/sat_pre_images",[7,9,11,16], normalize=True)

    # Load measured precipitation to create Y (output of the network)
    y = np.load("../dataset/sat_pre_images/crsflux_30.npy")[:,:,:,np.newaxis]
    y = np.log(1+y)
    # Verify dimensions of the data.
    # We should have 1163 samples of 400x400 images for both X and Y
    print(x.shape, y.shape)

    # Instantiate model defined in function above
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess)
    #K.tensorflow_backend._get_available_gpus()
    model = GetModel()

    # Fit data using a 70/30 validation split
    history = model.fit(x, y, epochs=20, verbose=1, validation_split=.3, shuffle=True)

    # Save the model once trained for later use
    #model.save('cnn_rain.h5')

    # Generate images from the data to see if model makes sense
    y_pred = model.predict(x[:1, :, :, :])
    plt.imsave("pred_rain_1_log.png", y_pred[0,:,:,0])
    plt.imsave("obs_rain_1_log.png", y[0,:,:,0])


if __name__ == "__main__":
    main()