import itertools

import keras
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

from constants import *

# layers = [
#     # part 1
#     Convolution2D(8, (kernel, kernel), padding='same', input_shape=(img_h, img_w, img_d)),
#     Activation('relu'),
#     Convolution2D(8, (kernel, kernel), padding='same'),
#     Activation('relu'),
#     MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     Convolution2D(16, (kernel, kernel), padding='same')
#     Activation('relu'),
#     Convolution2D(16, (kernel, kernel), padding='same'),
#     Activation('relu'),
#     MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     Convolution2D(32, (kernel, kernel), padding='same')
#     Activation('relu'),
#     Convolution2D(32, (kernel, kernel), padding='same'),
#     Activation('relu'),
#     MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     Convolution2D(64, (kernel, kernel), padding='same'),
#     Activation('relu'),
#     # Convolution2D(64, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # Convolution2D(128, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # Convolution2D(128, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # Convolution2D(256, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # Convolution2D(256, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # MaxPooling2D((2, 2)), # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # Convolution2D(512, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # Convolution2D(512, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     # Convolution2D(256, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # Convolution2D(256, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     # Convolution2D(128, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # Convolution2D(128, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     # Convolution2D(64, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     # Convolution2D(64, (kernel, kernel), padding='same'),
#     # Activation('relu'),
#     UpSampling2D((2, 2)), # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     Convolution2D(32, (kernel, kernel), padding='same'),
#     Activation('relu'),
#     Convolution2D(32, (kernel, kernel), padding='same'),
#     Activation('relu'),

#     # part 3
#     Convolution2D(n_labels, (1, 1), padding='valid'),
#     Reshape((img_h, img_w)),
#     # Permute((2, 3, 1))
#     Activation('softmax')
# ]


LAYERS = '8_8_16_16_32_32_16_16_8_8'
LAYERS = '16_16_32_32_16_16'
LAYERS = '16_32_16'
LAYERS = [int(s) for s in LAYERS.split('_')]
def _map(prev, cur, next):
    if prev is None:
        print('  ** beg')
        print('  ** conv', cur)
        yield Convolution2D(cur, (kernel, kernel), padding='same', input_shape=(img_h, img_w, img_d))
        yield Activation('relu')
        return
    elif prev < cur:
        print('  ** x2')
        yield MaxPooling2D((2, 2))
    elif prev > cur:
        print('  ** /2')
        yield UpSampling2D((2, 2))
    print('  ** conv', cur)
    yield Convolution2D(cur, (kernel, kernel), padding='same')
    yield Activation('relu')
    if next is None:
        print('  ** end')
        yield Convolution2D(n_labels, (1, 1), padding='valid')
        yield Reshape((img_h, img_w))
        yield Activation('softmax')


def create_model():
    layers = list(itertools.chain.from_iterable((
        _map(prev, cur, next)
        for (prev, cur, next) in zip([None] + LAYERS[:-1], LAYERS, LAYERS[1:] + [None])
    )))
    autoencoder = keras.models.Sequential()
    for l in layers:
        autoencoder.add(l)
    autoencoder.compile(loss="binary_crossentropy", optimizer='adadelta', metrics=[
        'accuracy', keras.metrics.binary_accuracy,
    ])
    return autoencoder
