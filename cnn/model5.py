from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras import backend as K
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.losses import mean_squared_error
import tensorflow as tf
import xarray as xr
from keras.callbacks import *

import numpy as np
from utils.IO import *

def closs(y_true, y_pred):
    l2 = mean_squared_error(y_true, y_pred)
    lgdl = squareSobelLoss(y_true, y_pred)
    # model1 0.8 0.2 model2 1 0 model3 0.5 0.5
    # return 0.8*l2+0.2*lgdl
    return 0.5*l2+0.5*lgdl

def expandedSobel(inputTensor):
    sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                              [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                              [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels


def squareSobelLoss(yTrue,yPred):
    # FROM:
    # https://stackoverflow.com/questions/47346615/how-to-implement-custom-sobel-filter-based-loss-function-using-keras
    #same beginning as the other loss
    filt = expandedSobel(yTrue)
    squareSobelTrue = K.square(K.depthwise_conv2d(yTrue,filt))
    squareSobelPred = K.square(K.depthwise_conv2d(yPred,filt))

    #here, since we've got 6 output channels (for an RGB image)
    #let's reorganize in order to easily sum X² and Y²: change (h,w,6) to (h,w,3,2)
    #caution: this method of reshaping only works in tensorflow
    #if you do need this in other backends, let me know
    newShape = K.shape(squareSobelTrue)
    newShape = K.concatenate([newShape[:-1],
                              newShape[-1:]//2,
                              K.variable([2],dtype='int32')])

    #sum the last axis (the one that is 2 above, representing X² and Y²)
    squareSobelTrue = K.sum(K.reshape(squareSobelTrue,newShape),axis=-1)
    squareSobelPred = K.sum(K.reshape(squareSobelPred,newShape),axis=-1)

    #since both previous values are already squared, maybe we shouldn't square them again?
    #but you can apply the K.sqrt() in both, and then make the difference,
    #and then another square, it's up to you...
    return K.mean(K.abs(squareSobelTrue - squareSobelPred))
def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(200, 200, 6)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, (3, 3), activation='relu', padding='same'))
    model.compile(loss=closs, optimizer=Adam(lr=1e-4), metrics=['mse'])
    print(model.summary())
    return model

def main():
    # Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
    x = load_dataset("../dataset/sat_pre_images",[7,9,11,16], normalize=True)[1:, :200, :200, :]
    ds = xr.open_dataset("../dataset/sat_pre_images/kmeans.nc")

    x = np.concatenate((x, ds['B7'].values[1:,:200,:200,np.newaxis]), axis=3)
    x = np.concatenate((x, ds['B9'].values[1:, :200, :200, np.newaxis]), axis=3)
    print(x.shape)

    # Load measured precipitation to create Y (output of the network)
    y = np.load("../dataset/sat_pre_images/crsflux_30.npy")[1:,:200,:200,np.newaxis]
    y = np.log(1+y)
    # Verify dimensions of the data.
    # We should have 1163 samples of 400x400 images for both X and Y
    print(x.shape, y.shape)

    # Instantiate model defined in function above
    model = GetModel()
    # Fit data using a 70/30 validation split

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('bestm53.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit(x, y, epochs=100, verbose=1, validation_split=.20, shuffle=True,callbacks=[es,mc])

    # Save the model once trained for later use
    model.save('model5_3.h5')

    # Generate images from the data to see if model makes sense
    y_pred = model.predict(x[:1, :, :, :])
    #plt.imsave("pred_rain_2.png", np.exp(y_pred[0,:,:,0]))
    #plt.imsave("obs_rain_2.png", np.exp(y[0,:,:,0]))
    plt.imsave("pred_rain_5_3_log.png", np.exp(y_pred[0, :, :, 0])-1)
    plt.imsave("obs_rain_5_3_log.png", np.exp(y[0,:,:,0])-1)



if __name__ == "__main__":
    main()