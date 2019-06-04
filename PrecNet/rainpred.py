"""
    rainpred.py

    Rain prediction network
"""

import xarray as xr
from keras.callbacks import *
from keras.layers import BatchNormalization, Conv2D
from keras.losses import mean_absolute_error
from keras.models import *
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from Sharpening.sharpening import *
from Clustering.clustering import process_batch_meanshift
import argparse

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



def loaddataset(dataset, clusters, prediction):
    # Load measured precipitation to create Y (output of the network)
    ds = xr.open_dataset(dataset)

    print(ds['B7'].shape)
    data = ds['B7'].shape[0]
    print(ds)
    images = np.zeros((data - 1, 200, 200, 6))
    images[:, :, :, 0] = ds['B7'].values[1:, 0:200, 0:200]
    print('loaded B7')
    images[:, :, :, 1] = ds['B9'].values[1:, 0:200, 0:200]
    print('loaded B9')
    images[:, :, :, 2] = ds['B11'].values[1:, 0:200, 0:200]
    print('loaded B11')
    images[:, :, :, 3] = ds['B16'].values[1:, 0:200, 0:200]
    print('loaded B16')

    if clusters is not None:
        ds = xr.open_dataset(clusters)
        print(ds['K7'].shape)
        images[:, :, :, 4] = ds['K7'].values[1:, 0:200, 0:200]
        print('loaded K7')
        images[:, :, :, 5] = ds['K11'].values[1:, 0:200, 0:200]
        print('loaded K11')
    else:
        images[:, :, :, 4] = process_batch_meanshift(images[:, :, :, 0])
        images[:, :, :, 5] = process_batch_meanshift(images[:, :, :, 2])
    
    
    ds2 = xr.open_dataset(prediction)
    print(ds2)
    y = ds2['TP'].values[1:, 0:200, 0:200, np.newaxis]
    print('loaded True Predictions')
    y2 = ds2['NP'].values[1:, 0:200, 0:200, np.newaxis]
    print('loaded Numerical Predictions')
    return images, y, y2


def train(dataset, clusters, prediction,output, modelfile, weights, val):

    images, y, y2 = loaddataset(dataset, clusters, prediction)

    # Verify dimensions of the data.
    # We should have 1163 samples of 400x400 images for both X and Y
    print(images.shape, y.shape)

    # Instantiate model defined in function above
    model = GetModel()
    # Fit data using a 70/30 validation split

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('best_'+weights, monitor='val_loss', mode='min', verbose=1, save_best_only=True)


    history = model.fit(images, y, epochs=40, verbose=1, validation_split=.20, shuffle=True, callbacks=[es, mc])
    # Save the model once trained for later use


    model.save(modelfile)
    # We load the best weights
    model.load_weights('best_'+weights)

    if val:
        y_pred = model.predict(images[:20, :, :, :])

        # y = np.exp(y)-1
        # y_pred = np.exp(y_pred)-1
        for i in range(20):
            # y_pred = np.exp(y_pred)-1
            frame = y_pred[i, :, :, 0]
            sharpened = sharpen(frame)

            plt.imsave("pred_rain_"+str(i)+"png", y_pred[i, :, :, 0])
            plt.imsave("sharpened_"+str(i)+"png", sharpened)
            plt.imsave("real_rain_"+str(i)+"png", y[i, :, :, 0])
            plt.imsave("WPS_rain__"+str(i)+"png", y2[i, :, :, 0])

    y_pred = model.predict(images[:, :, :, :])
    ds = xr.Dataset({})

    sharpen = sharpen_batch(y_pred[:,:,:,0])
    ds['Pred'] = xr.DataArray(y_pred[:,:,:,0], dims=['time', 'width', 'height'])
    ds['Real'] = xr.DataArray(y[:,:,:,0], dims=['time', 'width', 'height'])
    ds['Num'] = xr.DataArray(y2[:,:,:,0], dims=['time', 'width', 'height'])
    ds['Shar'] = xr.DataArray(sharpen[:, :, :], dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    print(ds)

    ds.to_netcdf(output, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)




def test(dataset, clusters, prediction, output, modelfile, weights):

    model = load_model(modelfile)
    model.load_weights(weights)

    ds = xr.open_dataset(dataset)
    print(ds['B7'].shape)
    data = ds['B7'].shape[0]
    print(ds)
    images = np.zeros((data - 1, 200, 200, 6))
    images[:, :, :, 0] = ds['B7'].values[1:, 0:200, 0:200]
    print('loaded B7')
    images[:, :, :, 1] = ds['B9'].values[1:, 0:200, 0:200]
    print('loaded B9')
    images[:, :, :, 2] = ds['B11'].values[1:, 0:200, 0:200]
    print('loaded B11')
    images[:, :, :, 3] = ds['B16'].values[1:, 0:200, 0:200]
    print('loaded B16')

    ds = xr.open_dataset(clusters)
    print(ds['K7'].shape)
    images[:, :, :, 4] = ds['K7'].values[1:, 0:200, 0:200]
    print('loaded K7')
    images[:, :, :, 5] = ds['K11'].values[1:, 0:200, 0:200]
    print('loaded K11')


    y_pred = model.predict(images[:, :, :, :])
    ds = xr.Dataset({})

    ds['Pred'] = xr.DataArray(y_pred[:,:,:,0], dims=['time', 'width', 'height'])

    if prediction is not None:
        ds2 = xr.open_dataset(prediction)
        print(ds2)
        y = ds2['TP'].values[1:, 0:200, 0:200, np.newaxis]
        print('loaded True Predictions')
        y2 = ds2['NP'].values[1:, 0:200, 0:200, np.newaxis]
        print('loaded Numerical Predictions')
        ds['Real'] = xr.DataArray(y[:,:,:,0], dims=['time', 'width', 'height'])
        ds['Num'] = xr.DataArray(y2[:,:,:,0], dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    print(ds)

    ds.to_netcdf(output, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)



if __name__ == "__main__":
    # Arguments:
    parser = argparse.ArgumentParser(description='Train/Test the rain prediction network')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    parser.add_argument('cluster',
                        help='cluster file of the dataset', type=str)
    parser.add_argument('--predictions',
                        help='Previous predictions to train', type=str, dest="pred", default=None)
    parser.add_argument('--netmodel',
                        help='Network model file', type=str, dest="nm",
                        required=False, default='STPN_model.h5')
    parser.add_argument('--netweight',
                        help='Network weights file', type=str, dest="nw",
                        required=False, default='STPN_weights.h5')
    parser.add_argument("--test", help="Perform testing (training by default) ", action="store_true", dest="test")
    parser.add_argument("--val", help="Perfoms testing on the dataset after training",
                        action="store_true",
                        dest="val")
    args = parser.parse_args()

    if args.test:
        test(args.input, args.cluster, args.pred, args.output, args.nm, args.nw)
    else:
        train(args.input, args.cluster, args.pred, args.output, args.nm, args.nw, args.val)
