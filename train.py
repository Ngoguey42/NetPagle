import keras
from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

import paths
from constants import *


c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)


encoding_layers = [
    Convolution2D(64, (kernel, kernel), padding='same', input_shape=(img_d, img_h, img_w)),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(64, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolution2D(128, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(128, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2, 2)),

    # Convolution2D(256, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2, 2)),

    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2, 2)),

    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2, 2)),
]

decoding_layers = [
    # UpSampling2D(),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),

    # UpSampling2D(),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),

    # UpSampling2D(),
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(128, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),

    # UpSampling2D(),
    # Convolution2D(128, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),
    # Convolution2D(64, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    # Activation('relu'),

    UpSampling2D(),
    Convolution2D(64, (kernel, kernel), padding='same'),
    # BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, (1, 1), padding='valid'),
    # BatchNormalization(),
]

def _create_model():
    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Permute((2, 3, 1)))
    autoencoder.add(Activation('softmax'))
    autoencoder.compile(loss="sparse_categorical_crossentropy", optimizer='adadelta', metrics=['accuracy'])
    return autoencoder

names = paths.create_names_list()
names = sorted(names)
x = [paths.img_of_name(fname) for fname in names]
y = [paths.mask_of_name(fname) for fname in names]

x = np.stack(x)
y = np.stack(y)
print('xy shapes   ', x.shape, y.shape)
print("Input data: xshape:{}, xsize:{}GB, yshape:{}, ymean{:%}".format(
    x.shape, x.size / 1024 ** 3,
    y.shape, y.mean(),
))


batch_size = 1

x = x.astype('uint8')
y = y.astype(bool)

model_path = paths.get_latest_model_path_opt()

if model_path is None:
    print('Creating model')
    m = _create_model()
else:
    print('Loading model:', model_path)
    m = models.load_model(model_path)

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self):
        super(ModelCheckpoint, self).__init__()
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        self.i += 1
        # if self.i % 6 != 0:
            # return
        loss = logs.get('loss', -42)
        acc = logs.get('acc', -42)

        path = paths.create_model_path(epoch, loss, acc)
        print("Saving to {} at epoch {} ({})".format(
            path, epoch, logs
        ))
        self.model.save(path)

print('Fiting')
cbs = [ModelCheckpoint()]
m.fit(x, y, epochs=10000, batch_size=batch_size, verbose=1, callbacks=cbs)

# path = paths.create_model_path()
# print('Saving', path)
# m.save(path)
