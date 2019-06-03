import numpy as np
import xarray as xr
from keras import backend as K
from keras.layers import ConvLSTM2D
from keras.layers import concatenate
from keras.layers.convolutional import Conv3D, Conv2D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_logarithmic_error, mean_squared_error
from keras.models import *
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks import *
import matplotlib.pyplot as plt
from keras.models import model_from_json
from utils import *

def closs(y_true, y_pred):
    
    lgdl = squareSobelLoss(y_true, y_pred)
    l2 = mean_squared_error(y_true, y_pred)


    return 0.6*l2+0.4*lgdl

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
    #let's reorganize in order to easily sum X2 and Y2, change (h,w,6) to (h,w,3,2)
    #caution: this method of reshaping only works in tensorflow
    #if you do need this in other backends, let me know
    newShape = K.shape(squareSobelTrue)
    newShape = K.concatenate([newShape[:-1],
                              newShape[-1:]//2,
                              K.variable([2],dtype='int32')])

    #sum the last axis (the one that is 2 above, representing X2 and Y2)
    squareSobelTrue = K.sum(K.reshape(squareSobelTrue,newShape),axis=-1)
    squareSobelPred = K.sum(K.reshape(squareSobelPred,newShape),axis=-1)

    #since both previous values are already squared, maybe we shouldn't square them again?
    #but you can apply the K.sqrt() in both, and then make the difference,
    #and then another square, it's up to you...
    return K.mean(K.abs(squareSobelTrue - squareSobelPred))


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

    
def main():
    #Load dataset

    # cap = xr.open_dataset("../dataset/sat_pre_video2/HIM8_2017.nc")  # H8_Flow.nc
    # # We need to intercept frames if using video2
    # cap2 = xr.open_dataset("../dataset/sat_pre_video2/TP_2017.nc")  # H8_Flow.nc
    # cap_time = cap.time[:].data
    # cap2_time = cap2.time[:].data
    # times = np.intersect1d(cap_time, cap2_time)
    # print(cap)
    #
    # time  = len(times-1)
    # images = np.zeros((time-1, 200, 200, 4))
    # images[:, :, :, 0] = cap.B7.sel(time=times)[:].data[:-1, 0:200, 0:200]
    # images[:, :, :, 1] = cap.B9.sel(time=times)[:].data[:-1, 0:200, 0:200]
    # images[:, :, :, 2] = cap.B11.sel(time=times)[:].data[:-1, 0:200, 0:200]
    # images[:, :, :, 3] = cap.B16.sel(time=times)[:].data[:-1, 0:200, 0:200]
    # print(images.shape)

    ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")
    print(ds)

    time = ds['B7'].shape[0]
    images = np.zeros((time - 1, 200, 200, 4))
    images[:, :, :, 0] = ds['B7'].values[:-1, 0:200, 0:200]
    images[:, :, :, 1] = ds['B9'].values[:-1, 0:200, 0:200]
    images[:, :, :, 2] = ds['B11'].values[:-1, 0:200, 0:200]
    images[:, :, :, 3] = ds['B16'].values[:-1, 0:200, 0:200]
    print(images.shape)


    in_frames= 3
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
    mc = ModelCheckpoint('bestSTPN_4_weights.h5',monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.fit_generator(MyGenerator_Train, epochs=100, verbose=1, shuffle=False,
                                  validation_data=MyGenerator_Test,
                                  use_multiprocessing=False, workers=4, callbacks=[es,mc])
    print('saving model')
    model.save('STPN_4_model.h5') 
    model_json = model.to_json()
    with open('STPN_4_model.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    model.save_weights('STPN_4_weights.h5')

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

    print('Loading best weights')
    model.load_weights('bestSTPN_4_weights.h5')
    X,Y = MyGenerator_Test.__getitem__(0)
    Y_pred = model.predict(X)
    print(Y_pred.shape)
    plt.imsave("pred_STPN_4_best.png", np.exp(Y_pred[7, :, :, 2])-1)
    print("predicting video")
    
    MyGenerator_video = Generator_Flow( xtrainID, ytrainID, images, batch_size=8)
    
    filename = "STPN_vid1.nc"
    ds = xr.Dataset({})
    
    res = np.zeros((time, 200,200,4))
    i = 0
    for X, Y in MyGenerator_video:
        frames = model.predict(X)
        for frame in frames:
                plt.imsave("frames4/STPN_3_B7"+str(i)+".png", frame[:, :, 0])
                plt.imsave("frames4/STPN_3_B9"+str(i)+".png", frame[:, :, 1])
                plt.imsave("frames4/STPN_3_B11"+str(i)+".png", frame[:, :, 2])
                plt.imsave("frames4/STPN_3_B16"+str(i)+".png", frame[:, :, 3])
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

def create_video():
    ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")
    print(ds)

    time = ds['B7'].shape[0]
    images = np.zeros((time - 1, 200, 200, 4))
    images[:, :, :, 0] = ds['B7'].values[:-1, 0:200, 0:200]
    images[:, :, :, 1] = ds['B9'].values[:-1, 0:200, 0:200]
    images[:, :, :, 2] = ds['B11'].values[:-1, 0:200, 0:200]
    images[:, :, :, 3] = ds['B16'].values[:-1, 0:200, 0:200]
    print(images.shape)


    in_frames = 3
    xtrainID, ytrainID = dataGenerator(time, in_frames, 1)
    MyGenerator_video = Generator_Flow(xtrainID, ytrainID, images, batch_size=1)

    model = load_model_json('STPN_3_model.json', 'bestSTPN_3_weights.h5')
    X,Y = MyGenerator_video.__getitem__(0)
    Y_pred = model.predict(X)
    print(Y_pred.shape)
    plt.imsave("pred_STPN_3_best.png", Y_pred[0, :, :, 2])
    print("predicting video")
    
    MyGenerator_video = Generator_Flow( xtrainID, ytrainID, images, batch_size=8)
    
    filename = "STPN_vid1.nc"
    ds = xr.Dataset({})
    
    res = np.zeros((time, 200,200,4))
    i = 0
    for X, Y in MyGenerator_video:
        frames = model.predict(X)
        for frame in frames:
                plt.imsave("frames3/STPN_3_B7"+str(i)+".png", frame[:, :, 0])
                plt.imsave("frames3/STPN_3_B9"+str(i)+".png", frame[:, :, 1])
                plt.imsave("frames3/STPN_3_B11"+str(i)+".png", frame[:, :, 2])
                plt.imsave("frames3/STPN_3_B16"+str(i)+".png", frame[:, :, 3])
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
    main()
    #create_video()
