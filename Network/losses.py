import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

image_shape = (200, 200, 3)

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true[:,:,:,0:3]) - loss_model(y_pred[:,:,:,0:3])))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)