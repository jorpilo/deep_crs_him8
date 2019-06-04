"""
    frame_pred.py

    Frame prediction network training and testing
"""

import argparse
import matplotlib.pyplot as plt
import xarray as xr
from keras.callbacks import *
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import *
from keras.optimizers import Adam
from sklearn.utils import shuffle
from losses import *
from utils import *

def load_model_json(pmodel, pweight):
    json_file = open(pmodel, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(pweight)
    return model


def save_mode_json(model, mf, wf):
    print('saving model')
    model_json = model.to_json()
    with open(mf, "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    model.save_weights(wf)


def GetModel(in_frames):

    im = Input((in_frames, 200, 200, 6), name="images_input")


    #STPN
    stpnconv = ConvLSTM2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last", return_sequences=True)(im)
    stpnconv = BatchNormalization(axis=-1)(stpnconv)
    stpnconv = ConvLSTM2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last", return_sequences=True)(stpnconv)
    stpnconv = BatchNormalization(axis=-1)(stpnconv)
    stpnconv = ConvLSTM2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last", return_sequences=False)(stpnconv)

    final = Conv2D(4, (5, 5), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(stpnconv)
    final = Conv2D(4, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(final)

    model = Model(inputs=[im], outputs=[final])
    model.compile(loss=closs, optimizer=Adam(lr=0.001), metrics=['msle'])

    print(model.summary())

    return model

    
def train(images, modelfile, weights, output, validate):
    #Load dataset

    in_frames=3
    xtrainID, ytrainID = dataGenerator(time, in_frames, 1)

    print(xtrainID.shape)
    print(ytrainID.shape)


    #Validation split
    print('shuffle ids')
    x_train, y_train = shuffle(xtrainID, ytrainID, random_state=0)

    print('split validation')
    number = int(np.round(x_train.shape[0]*0.25))
    x_test = x_train[0:number]
    y_test = y_train[0:number]

    x_train = x_train[number:-1]
    y_train = y_train[number:-1]


    print("Creating generators")
    MyGenerator_Train = Generator_Flow(x_train, y_train, images, batch_size=8)
    MyGenerator_Test = Generator_Flow(x_test, y_test, images, batch_size=8)

    model = GetModel(in_frames+1)
    print("Training network")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint("best_"+weights,monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.fit_generator(MyGenerator_Train, epochs=100, verbose=1, shuffle=False,
                                  validation_data=MyGenerator_Test,
                                  use_multiprocessing=False, workers=4, callbacks=[es,mc])

    save_mode_json(model, modelfile, weights)

    if validate:
        print('Loading best weights')
        model.load_weights('bestSTPN_4_weights.h5')

        print('Testing')
        X,Y = MyGenerator_Test.__getitem__(0)
        Y_pred = model.predict(X)
        print(Y_pred.shape)
        plt.imsave("readX1_STPN_4.png", X[7,0,:,:,2])
        plt.imsave("readX2_STPN_4.png", X[7,1,:,:,2])
        plt.imsave("readX3_STPN_4.png", X[7,2,:,:,2])
        plt.imsave("readX4_STPN_4.png", X[7,3,:,:,2])
        plt.imsave("pred_STPN_4.png", Y_pred[7, :, :, 2])
        plt.imsave("real_STPN_4.png", Y[7, :, :,2])

        X,Y = MyGenerator_Test.__getitem__(0)
        Y_pred = model.predict(X)
        print(Y_pred.shape)
        plt.imsave("pred_STPN_4_best.png", np.exp(Y_pred[7, :, :, 2])-1)
        print("predicting video")

    MyGenerator_video = Generator_Flow(xtrainID, ytrainID, images, batch_size=8)

    filename = output
    ds = xr.Dataset({})

    res = np.zeros((time, 200,200,4))
    i = 0
    for X, Y in MyGenerator_video:
        frames = model.predict(X)
        for frame in frames:
                res[i, :,:,:]= frame[:, :, :]
                i = i+1

    ds['B7'] = xr.DataArray(res[:,:,:,0], dims=['time', 'width', 'height'])
    ds['B9'] = xr.DataArray(res[:,:,:,1], dims=['time', 'width', 'height'])
    ds['B11'] = xr.DataArray(res[:,:,:,2], dims=['time', 'width', 'height'])
    ds['B16'] = xr.DataArray(res[:,:,:,3], dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    print(ds)

    ds.to_netcdf(filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)

    print('Done')

def test(images, modelfile, weights, output):

    in_frames = 3
    xtrainID, ytrainID = dataGenerator(time, in_frames, 1)

    model = load_model_json(modelfile, weights)

    print("predicting video")
    
    MyGenerator_video = Generator_Flow(xtrainID, ytrainID, images, batch_size=8)
    
    filename = output
    ds = xr.Dataset({})
    
    res = np.zeros((time, 200,200,4))
    i = 0
    for X, Y in MyGenerator_video:
        frames = model.predict(X)
        for frame in frames:
                res[i, :,:,:]= frame[:, :, :]
                i = i+1
    
    ds['B7'] = xr.DataArray(res[:,:,:,0], dims=['time', 'width', 'height'])
    ds['B9'] = xr.DataArray(res[:,:,:,1], dims=['time', 'width', 'height'])
    ds['B11'] = xr.DataArray(res[:,:,:,2], dims=['time', 'width', 'height'])
    ds['B16'] = xr.DataArray(res[:,:,:,3], dims=['time', 'width', 'height'])
    
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf(filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical Flow algorithm')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    parser.add_argument('--netmodel',
                        help='Network model file', type=str, dest="nm",
                        required=False, default='STPN_model.h5')
    parser.add_argument('--netweight',
                        help='Network weights file', type=str, dest="nw",
                        required=False, default='STPN_weights.h5')
    parser.add_argument("-t", "--test", help="Perform testing (training by default) ", action="store_true", dest="test")
    parser.add_argument("-a", "--validate", help="Perfoms testing on the dataset after training", action="store_true",
                        dest="val")

    args = parser.parse_args()
    args.input = "../dataset/sat_prec_vid1"
    ds = xr.open_dataset(args.input)  # H8_Flow.nc

    time = ds['B7'].shape[0]
    images = np.zeros((time - 1, 200, 200, 4))
    images[:, :, :, 0] = ds['B7'].values[:-1, 0:200, 0:200]
    images[:, :, :, 1] = ds['B9'].values[:-1, 0:200, 0:200]
    images[:, :, :, 2] = ds['B11'].values[:-1, 0:200, 0:200]
    images[:, :, :, 3] = ds['B16'].values[:-1, 0:200, 0:200]

    if args.test:
        test(images, args.nm, args.nw, args.output)
    else:
        train(images, args.nm, args.nw, args.output, args.val)
