import keras
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

from constants import *
layers = [
    # part 1
    Convolution2D(64, (kernel, kernel), padding='same', input_shape=(img_h, img_w, img_d)),
    Activation('relu'),
    # Convolution2D(32, (kernel, kernel), padding='same'),
    # Activation('relu'),
    # MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Convolution2D(64, (kernel, kernel), padding='same'),
    # Activation('relu'),
    Convolution2D(64, (kernel, kernel), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Convolution2D(128, (kernel, kernel), padding='same'),
    Activation('relu'),
    Convolution2D(128, (kernel, kernel), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Convolution2D(256, (kernel, kernel), padding='same'),
    Activation('relu'),
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # Activation('relu'),
    # MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # Activation('relu'),
    # Convolution2D(512, (kernel, kernel), padding='same'),
    # Activation('relu'),
    # UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Convolution2D(256, (kernel, kernel), padding='same'),
    # Activation('relu'),
    Convolution2D(256, (kernel, kernel), padding='same'),
    Activation('relu'),
    UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Convolution2D(128, (kernel, kernel), padding='same'),
    Activation('relu'),
    Convolution2D(128, (kernel, kernel), padding='same'),
    Activation('relu'),
    UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Convolution2D(64, (kernel, kernel), padding='same'),
    Activation('relu'),
    Convolution2D(64, (kernel, kernel), padding='same'),
    Activation('relu'),
    # UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Convolution2D(32, (kernel, kernel), padding='same'),
    # Activation('relu'),
    # Convolution2D(32, (kernel, kernel), padding='same'),
    # Activation('relu'),

    # part 3
    Convolution2D(n_labels, (1, 1), padding='valid'),
    Reshape((img_h, img_w)),
    # Permute((2, 3, 1))
    Activation('softmax')
]

def create_model():
    autoencoder = keras.models.Sequential()
    for l in layers:
        autoencoder.add(l)
    autoencoder.compile(loss="binary_crossentropy", optimizer='adadelta', metrics=[
        'accuracy', keras.metrics.binary_accuracy,
    ])
    return autoencoder
