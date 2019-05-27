import xarray as xr
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Conv3D, Conv2D, Conv3DTranspose, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import AveragePooling3D
from utils import *
from sklearn.utils import shuffle
from keras.models import *
import matplotlib.pyplot as plt
from keras.layers import Lambda


def GetModel(in_frames):
    inputs = Input((in_frames, 256, 256, 3))
    B1 = BatchNormalization(axis=4)(inputs)
    C1 = Conv3D(16, (in_frames, 5, 5), strides=(1, 1, 1), activation='relu', padding='same',  data_format="channels_last")(B1)
    B2 = BatchNormalization(axis=4)(C1)
    C2 = Conv3D(32, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(B2)
    B3 = BatchNormalization(axis=4)(C2)
    C3 = Conv3D(64, (in_frames, 5, 5), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(B3)
    B4 = BatchNormalization(axis=4)(C3)
    C4 = Conv3DTranspose(32, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(B4)
    B5 = BatchNormalization(axis=4)(C4)
    C5 = Conv3DTranspose(16, (in_frames, 5, 5), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(B5)
    Av = AveragePooling3D(pool_size=(in_frames, 1, 1), strides=None, padding='same', data_format='channels_last')(C5)
    sAv = Lambda(lambda x: x[:, 0, :, :,:])(Av)
    C6 = Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(sAv)
    C7 = Conv2D(3, (5, 5), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(C6)
    C8 = Conv2D(3, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(C7)
    model = Model(inputs=[inputs], outputs=[C8])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01), metrics=['mse'])

    print(model.summary())

    return model


def main():
    #Load dataset
    ds = xr.open_dataset("../dataset/sat_pre_video1/opticalflow.nc")
    #ds2 = xr.open_dataset("../dataset/sat_pre_video2/opticalflow.nc")
    ds1len = len(ds['sin'])
    total = 2*ds1len
    #total = 2*ds1len+len(ds2['sin'])

    #w = ds2['sin'].shape[1]
    #h = ds2['sin'].shape[2]
    all = np.zeros((total, 200, 200, 2))

    print("loading sin")
    val = ds['sin'].values
    all[0:ds1len, :,:,0]= val[:,0:200, 0:200]
    all[ds1len:2*ds1len, :, :, 0] = val[:,-200:, -200:]
    #val = ds2['sin'].values
    #all[2*ds1len:, :,:, 0] = val

    print("loading cos")
    val = ds['cos'].values
    all[0:ds1len, :, :, 1] = val[:, 0:200, 0:200]
    all[ds1len:2 * ds1len, :, :, 1] = val[:, -200:, -200:]
    #val = ds2['cos'].values
    #all[2 * ds1len:, :,:, 1] = val

    print("loading mag")
    val = ds['mag'].values
    all[0:ds1len, :, :, 2] = val[:, 0:200, 0:200]
    all[ds1len:2 * ds1len, :, :, 2] = val[:, -200:, -200:]
    #val = ds2['mag'].values
    #all[2 * ds1len:, :,:, 2] = val

    print(all.shape)

    x_train, y_train = dataGenerator(all,learning_frames=2, steps=1)
    print(x_train.shape)
    print(y_train.shape)

    #Validation split
    print("Shuffling data")
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    print("Splitting data")
    number = int(np.round(x_train.shape[0]*0.25))
    x_test = x_train[0:number]
    y_test = y_train[0:number]

    x_train = x_train[number:-1]
    y_train = y_train[number:-1]
    print("Creating generators")
    MyGenerator_Train = Generator(x_train,y_train, all, batch_size=20)
    MyGenerator_Test = Generator(x_test, y_test, all, batch_size=20)

    model = GetModel(in_frames=2)
    print("Training network")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('OPFN_weights.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                         save_weights_only=True)
    model.fit_generator(MyGenerator_Train, epochs=60,verbose=1, shuffle=False,
                                  validation_data=MyGenerator_Test,
                                  use_multiprocessing=True, workers=8, callbacks=[es,mc])
    print('saving model')
    model_json = model.to_json()
    with open('OPFN_model.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    model.save_weights('OPFN_weights.h5')

if __name__ == "__main__":
    main()