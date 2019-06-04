
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, LeakyReLU
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import mean_squared_logarithmic_error
from matplotlib import pyplot as plt
from keras.losses import mean_squared_error
import tensorflow as tf
import xarray as xr
from keras.callbacks import *
from keras.models import *
import keras.losses
import numpy as np
from utils.IO import *
import sys
sys.path.append("..")
from segmentation.kmeans_transform_files import *


HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x= K.abs(y_true - y_pred)
   x= K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)


def closs(y_true, y_pred):
    lgdl = squareSobelLoss(y_true, y_pred)
    l2 = mean_squared_error(y_true, y_pred)
    l1 = smoothL1(y_true, y_pred)
    return l1+l2 + lgdl


def expandedSobel(inputTensor):
    sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                              [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                              [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])

    # this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0, 0, 0, :]), (1, 1, -1, 1))
    # if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels


def squareSobelLoss(yTrue, yPred):
    # FROM:
    # https://stackoverflow.com/questions/47346615/how-to-implement-custom-sobel-filter-based-loss-function-using-keras
    # same beginning as the other loss
    filt = expandedSobel(yTrue)
    squareSobelTrue = K.square(K.depthwise_conv2d(yTrue, filt))
    squareSobelPred = K.square(K.depthwise_conv2d(yPred, filt))

    # here, since we've got 6 output channels (for an RGB image)
    # let's reorganize in order to easily sum X2 and Y2, change (h,w,6) to (h,w,3,2)
    # caution: this method of reshaping only works in tensorflow
    # if you do need this in other backends, let me know
    newShape = K.shape(squareSobelTrue)
    newShape = K.concatenate([newShape[:-1],
                              newShape[-1:] // 2,
                              K.variable([2], dtype='int32')])

    # sum the last axis (the one that is 2 above, representing X2 and Y2)
    squareSobelTrue = K.sum(K.reshape(squareSobelTrue, newShape), axis=-1)
    squareSobelPred = K.sum(K.reshape(squareSobelPred, newShape), axis=-1)

    # since both previous values are already squared, maybe we shouldn't square them again?
    # but you can apply the K.sqrt() in both, and then make the difference,
    # and then another square, it's up to you...
    return K.mean(K.abs(squareSobelTrue - squareSobelPred))


def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(200, 200, 4)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    model.compile(loss=closs, optimizer=Adam(lr=0.005), metrics=['mse'])
    print(model.summary())
    return model

def main():
    # Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
    #x = load_dataset("../dataset/sat_pre_images",[7,9,11,16], normalize=True)[1:, :200, :200, :]
    #ds = xr.open_dataset("../dataset/sat_pre_images/kmeans.nc")

    cap = xr.open_dataset("../dataset/sat_pre_video2/HIM8_2017.nc")  # H8_Flow.nc
    # We need to intercept frames if using video2
    cap2 = xr.open_dataset("../dataset/sat_pre_video2/TP_2017.nc")  # H8_Flow.nc
    cap3 = xr.open_dataset("MeanshiftVid2.nc")  # H8_Flow.nc
    cap_time = cap.time[:].data
    cap2_time = cap2.time[:].data
    times = np.intersect1d(cap_time, cap2_time)
    print(cap)

    time = 1500
    images = np.zeros((time, 200, 200, 4))
    images[:, :, :, 0] = cap.B7.sel(time=times[3500:3500+time])[:].data[:, 0:200, 0:200]
    print('loaded B7')
    images[:, :, :, 1] = cap.B9.sel(time=times[3500:3500+time])[:].data[:, 0:200, 0:200]
    print('loaded B9')
    images[:, :, :, 2] = cap.B11.sel(time=times[3500:3500+time])[:].data[:, 0:200, 0:200]
    print('loaded B11')
    images[:, :, :, 3] = cap.B16.sel(time=times[3500:3500+time])[:].data[:, 0:200, 0:200]
    print('loaded B16')
    print(images.shape)

    # images[:, :, :, 4] = cap3['K7'].values
    #
    # images[:, :, :, 5] = cap3['K11'].values
    # print('loaded K11')
    x= images
    print(x.shape)

    # process = process_batch_meanshift(images[:,:200,:200,0])
    # images[:, :, :, 4] = process
    # print('loaded K7')
    # process = process_batch_meanshift(images[:,:200,:200,2])
    # images[:, :, :, 5] = process
    # print('loaded K11')
    # x= images
    # print(x.shape)
    #
    # ds = xr.Dataset({})
    # ds['K7'] = xr.DataArray(images[:, :, :, 4], dims=['time', 'width', 'height'])
    # ds['K11'] = xr.DataArray(images[:, :, :, 5], dims=['time', 'width', 'height'])
    # comp = dict(zlib=True, complevel=9)
    # encoding = {var: comp for var in ds.data_vars}
    #
    # ds.to_netcdf('MeanshiftVid2.nc', mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)
    #
    # # Load measured precipitation to create Y (output of the network)
    y = cap2.tp.sel(time=times[3500:3500+time])[:].data[:, 0:200, 0:200, np.newaxis]
    # y = np.log(1+y)
    # Verify dimensions of the data.
    # We should have 1163 samples of 400x400 images for both X and Y
    print(x.shape, y.shape)

    # Instantiate model defined in function above
    model = GetModel()
    # Fit data using a 70/30 validation split

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('bestm53.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit(x, y, epochs=60, verbose=1, validation_split=.3, shuffle=True,callbacks=[es,mc])

    # Save the model once trained for later use
    model.save('model5_3.h5')
    model.load_weights('bestm53.h5')
    # Generate images from the data to see if model makes sense
    y_pred = model.predict(x[:20, :, :, :])

    # y = np.exp(y)-1
    # y_pred = np.exp(y_pred)-1
    for i in range(20):
        plt.imsave("test5_1/pred_rain_"+str(i)+".png", y_pred[i,:,:,0])
        plt.imsave("test5_1/real_rain_" + str(i) + ".png", y[i,:,:,0])
        plt.imsave("test5_1/B7_" + str(i) + ".png", x[i, :, :, 0])

    Y = model.predict(x)
    ds = xr.Dataset({})
    data = Y[:, :, :, 0]
    # data = np.exp(data)-1
    ds['Pred'] = xr.DataArray(data, dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf('prediction_vid2.nc', mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

    # #Second part Video prediction
    # ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")
    #
    # len = ds['B7'].shape[0]
    # x = np.zeros((len, 200, 200, 6))
    #
    # x[:, :, :, 0] = ds['B7'].data[:, 0:200, 0:200]
    # print('loaded B7')
    # x[:, :, :, 1] = ds['B9'].data[:, 0:200, 0:200]
    # print('loaded B9')
    # x[:, :, :, 2] = ds['B11'].data[:, 0:200, 0:200]
    # print('loaded B11')
    # x[:, :, :, 3] = ds['B16'].data[:, 0:200, 0:200]
    # print('loaded B16')
    # x[:, :, :, 4] = process_batch_meanshift(x[:, :, :, 0])
    # print('loaded K7')
    # x[:, :, :, 5] = process_batch_meanshift(x[:, :, :, 2])
    # print('loaded K9')
    #
    # ds = xr.Dataset({})
    # ds['K7'] = xr.DataArray(x[:, :, :, 4], dims=['time', 'width', 'height'])
    # ds['K11'] = xr.DataArray(x[:, :, :, 5] , dims=['time', 'width', 'height'])
    # comp = dict(zlib=True, complevel=9)
    # encoding = {var: comp for var in ds.data_vars}
    #
    # ds.to_netcdf('MeanshiftVid1.nc', mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)
    #
    #
    # Y = model.predict(x)
    # ds = xr.Dataset({})
    # data = Y[:, :, :, 0]
    # ds['Pred'] = xr.DataArray(data, dims=['time', 'width', 'height'])
    #
    # comp = dict(zlib=True, complevel=9)
    # encoding = {var: comp for var in ds.data_vars}
    #
    # ds.to_netcdf('prediction_vid1.nc', mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

def test():

    keras.losses.closs = closs

    model = load_model('model5_3.h5')
    model.load_weights('bestm53.h5')
    ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")

    len = ds['B7'].shape[0]
    X = np.zeros((len, 200,200,6))

    X[:, :, :, 0]= ds['B7'].data[:,200:400,200:400]
    print('loaded B7')
    X[:, :, :, 1] = ds['B9'].data[:, 200:400,200:400]
    print('loaded B9')
    X[:, :, :, 2] = ds['B11'].data[:, 200:400,200:400]
    print('loaded B11')
    X[:, :, :, 3] = ds['B16'].data[:, 200:400, 200:200]
    print('loaded B16')
    X[:, :, :, 4] = process_batch(X[:, :, :, 0])
    print('loaded K7')
    X[:, :, :, 5] = process_batch(X[:, :, :, 1])
    print('loaded K9')

    Y = model.predict(X)
    plt.imshow(Y[0, :, :, 0])
    plt.show()

    ds = xr.Dataset({})
    ds['Pred'] = xr.DataArray(Y[:, :, :, 0], dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf('prediction_vid1.nc', mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)


if __name__ == "__main__":
    main()
    #test()