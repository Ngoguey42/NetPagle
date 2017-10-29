import keras
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

from constants import *

encoding_layers = [
    Convolution2D(8, (kernel, kernel), padding='same', input_shape=(img_h, img_w, img_d)),
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
layers = [
    # part 1
    Convolution2D(32, (kernel, kernel), padding='same', input_shape=(img_h, img_w, img_d)),
    Activation('relu'),
    Convolution2D(32, (kernel, kernel), padding='same'),
    Activation('relu'),

    MaxPooling2D((4, 4)),

    Convolution2D(128, (kernel, kernel), padding='same'),
    Activation('relu'),
    Convolution2D(128, (kernel, kernel), padding='same'),
    Activation('relu'),

    UpSampling2D((4, 4)),

    Convolution2D(32, (kernel, kernel), padding='same'),
    Activation('relu'),
    Convolution2D(32, (kernel, kernel), padding='same'),
    Activation('relu'),

    # part 3
    Convolution2D(n_labels, (1, 1), padding='valid'),
    Reshape((img_h, img_w)),
    # Permute((2, 3, 1))
    Activation('softmax')
]

def _create_model():
    autoencoder = keras.models.Sequential()
    # autoencoder.encoding_layers = encoding_layers
    # for l in autoencoder.encoding_layers:
    #     autoencoder.add(l)
    # autoencoder.decoding_layers = decoding_layers
    # for l in autoencoder.decoding_layers:
    #     autoencoder.add(l)
    for l in layers:
        autoencoder.add(l)
    autoencoder.compile(loss="binary_crossentropy", optimizer='adadelta', metrics=['accuracy'])
    # autoencoder.compile(loss="sparse_categorical_crossentropy", optimizer='adadelta', metrics=['accuracy'])
    return autoencoder
