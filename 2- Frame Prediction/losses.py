import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.losses import mean_squared_error
from keras.models import Model

image_shape = (200, 200, 3)

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true[:,:,:,0:3]) - loss_model(y_pred[:,:,:,0:3])))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def closs(y_true, y_pred):
    lgdl = squareSobelLoss(y_true, y_pred)
    l2 = mean_squared_error(y_true, y_pred)

    return 0.6 * l2 + 0.4 * lgdl


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