import numpy as np
import xarray as xr
from keras import backend as K
from keras.layers import ConvLSTM2D
from keras.layers import concatenate
from keras.layers.convolutional import Conv3D, Conv2D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.models import *
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks import *
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Flatten, LeakyReLU, Dropout
from utils import *
from losses import *


def closs(y_true, y_pred):
    l2 = mean_squared_error(y_true, y_pred)
    lgdl = squareSobelLoss(y_true, y_pred)

    return 0.8*l2+0.2*lgdl

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

def Discriminator():
    input = Input((200,200,4))
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = (LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(256, kernel_size=5, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[out], name='Discriminator')
    print(model.summary())
    return model

def Generator():
    in_frames = 3
    im = Input((in_frames, 200, 200, 4), name="images_input")
    of = Input((in_frames, 200, 200, 3), name="opticalFlow_input")

    #MEN

    ofconv = Conv3D(32, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same',  data_format="channels_last")(of)
    ofconv = BatchNormalization(axis=-1)(ofconv)
    ofconv = Conv3D(64, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(ofconv)
    ofconv = BatchNormalization(axis=-1)(ofconv)
    ofconv = Conv3DTranspose(32, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same',data_format="channels_last")(ofconv)
    ofconv = BatchNormalization(axis=-1)(ofconv)
    ofconv = Conv3DTranspose(16, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same',data_format="channels_last")(ofconv)

    #Image Transformation
    imconv = Conv3D(32, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(im)
    imconv = BatchNormalization(axis=-1)(imconv)
    imconv = Conv3D(64, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', data_format="channels_last")(imconv)
    imconv = BatchNormalization(axis=-1)(imconv)
    imconv = Conv3DTranspose(32, (in_frames, 3, 3), strides=(1, 1, 1), activation='relu', padding='same',data_format="channels_last")(imconv)

    #Concat

    catdata = concatenate([imconv, ofconv], axis=4)
    catdata = BatchNormalization(axis=-1)(catdata)

    #STPN

    stpnconv = ConvLSTM2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last", return_sequences=True)(catdata)
    stpnconv = BatchNormalization(axis=-1)(stpnconv)
    stpnconv = ConvLSTM2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last", return_sequences=True)(stpnconv)
    stpnconv = BatchNormalization(axis=-1)(stpnconv)
    stpnconv = ConvLSTM2D(4, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last", return_sequences=False)(stpnconv)

    final = Conv2D(4, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(stpnconv)
    final = Conv2D(4, (5, 5), strides=(1, 1), activation='relu', padding='same', data_format="channels_last")(final)

    model = Model(inputs=[im, of], outputs=[final], name='Generator')

    print(model.summary())

    return model

def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=(3,200,200,4))
    inputs2 = Input(shape=(3,200,200,3))
    generated_images = generator([inputs, inputs2])
    outputs = discriminator(generated_images)
    model = Model(inputs=[inputs, inputs2], outputs=[generated_images, outputs])
    return model

def main():
    #Load dataset


    ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")
    print(ds)

    time  = ds['B7'].shape[0]
    images = np.zeros((time-1, 200, 200, 4))
    images[:, :, :, 0] = ds['B7'].values[:-1, 0:200, 0:200]
    images[:, :, :, 1] = ds['B9'].values[:-1, 0:200, 0:200]
    images[:, :, :, 2] = ds['B11'].values[:-1, 0:200, 0:200]
    images[:, :, :, 3] = ds['B16'].values[:-1, 0:200, 0:200]
    print(images.shape)

    ds = xr.open_dataset('../dataset/sat_pre_video1/opticalflow.nc')
    print(ds)
    time = ds['sin'].shape[0]
    opticalflow = np.zeros((time, 200,200,3))
    opticalflow[:, :, :, 0] = ds['sin'].values[:,0:200,0:200]
    opticalflow[:, :, :, 1] = ds['cos'].values[:, 0:200, 0:200]
    opticalflow[:, :, :, 2] = ds['mag'].values[:, 0:200, 0:200]

    print(opticalflow.shape)

    in_frames= 3
    xtrainID, ytrainID = dataGenerator(time, in_frames, 1)

    print(xtrainID.shape)
    print(ytrainID.shape)


    #OFPN = load_model('my_model.h5')


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
    MyGenerator_Train = Generator2D(x_train, y_train, images, opticalflow, batch_size=8)
    MyGenerator_Test = Generator2D(x_test, y_test, images, opticalflow, batch_size=8)

    print("Training network")

    g = Generator()
    d = Discriminator()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
    g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    epoch_num = 100
    for epoch in range(epoch_num):
        for index in range(len(MyGenerator_Train)):
            X, Y = MyGenerator_Train.__getitem__(index)
            # Generate fake inputs
            generated_images = g.predict(X)
            plt.imsave("pred_STPN3.png", np.exp(generated_images[0,:,:,0])-1)
            # Train multiple times discriminator on real and fake inputs
            d_loss_real = d.train_on_batch(X[0], np.ones(8,1))
            d_loss_fake = d.train_on_batch(generated_images, K.zeros(8,1))
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            d.trainable = False
            # Train generator only on discriminator's decision and generated images
            d_on_g_loss = d_on_g.train_on_batch(X, [Y, K.ones(8,1)])

            d.trainable = True


    # print('saving model')
    # model_json = model.to_json()
    # with open('STPN_model.json', "w") as json_file:
    #     json_file.write(model_json)
    # json_file.close()
    # model.save_weights('STPN_weights.h5')

    print('Testing')
    X,Y = MyGenerator_Test.__getitem__(0)
    Y_pred = model.predict(X)
    print(Y_pred.shape)
    print(Y_pred[0, :, :, 2])
    print(Y[0, :, :,2])
    plt.imsave("pred_STPN.png", np.exp(Y_pred[0, :, :, 2]) - 1)
    plt.imsave("real_STPN.png", np.exp(Y[0, :, :,2]) - 1)

def predict_frame():
    ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")
    print(ds)

    time = ds['B7'].shape[0]
    images = np.zeros((time - 1, 200, 200, 4))
    images[:, :, :, 0] = ds['B7'].values[:-1, 0:200, 0:200]
    images[:, :, :, 1] = ds['B9'].values[:-1, 0:200, 0:200]
    images[:, :, :, 2] = ds['B11'].values[:-1, 0:200, 0:200]
    images[:, :, :, 3] = ds['B16'].values[:-1, 0:200, 0:200]
    print(images.shape)

    ds = xr.open_dataset('../dataset/sat_pre_video1/opticalflow.nc')
    print(ds)
    time = ds['sin'].shape[0]
    opticalflow = np.zeros((time, 200, 200, 3))
    opticalflow[:, :, :, 0] = ds['sin'].values[:, 0:200, 0:200]
    opticalflow[:, :, :, 1] = ds['cos'].values[:, 0:200, 0:200]
    opticalflow[:, :, :, 2] = ds['mag'].values[:, 0:200, 0:200]

    print(opticalflow.shape)

    xtrainID, ytrainID = dataGenerator(time, in_frames, 1)
    MyGenerator_Train = Generator2D(xtrainID, ytrainID, images, opticalflow, batch_size=1)

    X,Y = MyGenerator_Train.__getitem__(0)



    model = load_model_json('STPN_model.json', 'bestSTPN.h5')

    Y_pred = model.predict(X)
    print(Y_pred.shape)
    print(Y_pred[0, :, :, 2])
    print(Y[0, :, :,2])
    plt.imsave("pred_STPN.png", np.exp(Y_pred[0, :, :, 2]) - 1)
    plt.imsave("real_STPN.png", np.exp(Y[0, :, :,2]) - 1)


if __name__ == "__main__":
    main()
    #predict_frame()  