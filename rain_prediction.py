from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D
from keras import backend as K
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.callbacks import *
import numpy as np
from keras.models import *
import xarray as xr
from scipy import ndimage
from keras.losses import mean_squared_error, mean_absolute_error
from kmeans_transform_files import *


def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(200, 200, 6)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    model.compile(loss=mean_absolute_error, optimizer=Adam(lr=1e-4), metrics=['mse'])
    print(model.summary())
    return model

def train(data):
    ds = xr.open_dataset("../dataset/sat_pre_images/imgs.nc")
    print(ds['B7'].shape)
    data = ds['B7'].shape[0]
    print(ds)
    images = np.zeros((data-1, 200, 200, 6))
    images[:, :, :, 0] = ds['B7'].values[1:, 0:200, 0:200]
    print('loaded B7')
    images[:, :, :, 1] = ds['B9'].values[1:, 0:200, 0:200]
    print('loaded B9')
    images[:, :, :, 2] = ds['B11'].values[1:, 0:200, 0:200]
    print('loaded B11')
    images[:, :, :, 3] = ds['B16'].values[1:, 0:200, 0:200]
    print('loaded B16')

    ds = xr.open_dataset("../dataset/sat_pre_images/kmeans3.nc")
    print(ds['K7'].shape)
    images[:, :, :, 4] = ds['K7'].values[1:, 0:200, 0:200]
    print('loaded K7')
    images[:, :, :, 5] = ds['K11'].values[1:, 0:200, 0:200]
    print('loaded K11')

    # Load measured precipitation to create Y (output of the network)
    y2 = np.load("../dataset/sat_pre_images/gpm_30.npy")[1:,:200,:200,np.newaxis]
    y = np.load("../dataset/sat_pre_images/crsflux_30.npy")[1:, :200, :200, np.newaxis]

    # Verify dimensions of the data.
    # We should have 1163 samples of 400x400 images for both X and Y
    print(images.shape, y.shape)

    # Instantiate model defined in function above
    model = GetModel()
    # Fit data using a 70/30 validation split

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('bestm53.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


    history = model.fit(images, y, epochs=40, verbose=1, validation_split=.20, shuffle=True, callbacks=[es, mc])

    # Save the model once trained for later use
    model.save('model2.h5')

    y_pred = model.predict(images[:20, :, :, :])

    # y = np.exp(y)-1
    # y_pred = np.exp(y_pred)-1
    for i in range(20):
        # y_pred = np.exp(y_pred)-1
        frame = y_pred[i, :, :, 0]

        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        # blurred = ndimage.gaussian_filter(frame,3)
        # filter_blurred = ndimage.gaussian_filter(blurred, 3)
        # alpha = 30
        # sharpened = blurred + alpha*(blurred-filter_blurred)
        sharpened = cv2.filter2D(frame, -1, kernel_sharpening)

        plt.imsave("model1/pred_rain_"+str(i)+"png", y_pred[i, :, :, 0])
        plt.imsave("model1/sharpened_"+str(i)+"png", sharpened)
        plt.imsave("model1/real_rain_"+str(i)+"png", y[i, :, :, 0])
        plt.imsave("model1/WPS_rain__"+str(i)+"png", y2[i, :, :, 0])

def test(model_path, data):

    model = load_model('model2.h5')
    # Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
    x = load_dataset("../dataset/sat_pre_images", [7, 9, 11, 16], normalize=True)[:, :, :, :]
    y = np.load("../dataset/sat_pre_images/crsflux_30.npy")[:, :, :, np.newaxis]
    y2 = np.load("../dataset/sat_pre_images/gpm_30.npy")[:, :, :, np.newaxis]
    y_pred = model.predict(x[:,:, :, :])

    for i in range(x.shape[0]):
        plt.imsave("test/pred_rain_2_"+str(i)+".png", np.exp(y_pred[i,:,:,0])-1)
        plt.imsave("test/real_rain_2_" + str(i) + ".png", y[i, :, :, 0])
        plt.imsave("test/B7_2_" + str(i) + ".png", x[i, :, :, 0])


    ds = xr.Dataset({})
    ds['X'] = xr.DataArray(x[:, :, :, 0], dims=['time', 'width', 'height'])
    ds['Ypred'] =  xr.DataArray(np.exp(y_pred[:,:,:,0])-1, dims=['time', 'width', 'height'])
    ds['Ytrue'] = xr.DataArray(x[:, :, :, 0], dims=['time', 'width', 'height'])
    ds['YOld'] = xr.DataArray(y2[:, :, :, 0], dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf('test/prediction_imgs.nc', mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

if __name__ == "__main__":
    train()
