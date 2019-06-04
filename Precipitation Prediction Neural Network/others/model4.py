
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from utils.IO import *

def unet(pretrained_weights = None,input_size = (400,400,4)):
    inputs = Input(input_size)
    # Size 400x400x3
    conv1 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # Size 200x200x3
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # Size 100x100x3
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # Size 50x50x3
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # Size 25x25x3
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)



    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv11 = Conv2D(1, 1,  activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

    model = Model(input=inputs, output=conv11)

    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error', metrics=['mse'])
    print(model.summary())

    if pretrained_weights:
        model.load_weights(pretrained_weights)

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
    model = unet()

    # Fit data using a 70/30 validation split
    history = model.fit(x, y, epochs=20, verbose=1, validation_split=.3, shuffle=True)

    # Save the model once trained for later use
    #model.save('cnn_rain.h5')

    # Generate images from the data to see if model makes sense
    y_pred = model.predict(x[:1, :, :, :])
    plt.imsave("pred_rain_4_log.png", y_pred[0,:,:,0])
    plt.imsave("obs_rain_4_log.png", y[0,:,:,0])


if __name__ == "__main__":
    main()