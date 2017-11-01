import sys
import os

import keras
import tensorflow as tf

import model
import data_source
from constants import *

c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)

ds = data_source.DataSource(PREFIX)
m = model.Model(os.path.join(PREFIX, sys.argv[1]), ds)
# exit()
m.fit()
