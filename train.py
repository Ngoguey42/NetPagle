# from keras import models
# from keras.layers.core import Activation, Reshape, Permute
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import SGD
# from keras import backend as K

import keras
import tensorflow as tf
import numpy as np

import paths
from constants import *
import model

# keras.backend.set_image_dim_ordering('th')
c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)

names = paths.create_names_list()
names = sorted(names)
print("{} names".format(len(names)))
names = names[:5]
print("{} names".format(len(names)))
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
    m = model._create_model()
else:
    print('Loading model:', model_path)
    m = keras.models.load_model(model_path)

m.summary()

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self):
        super(ModelCheckpoint, self).__init__()
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        # exit()
        # return

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
