from pprint import pprint
import yaml

import numpy as np

import paths
from constants import *

# names = paths.create_names_list()
# names = sorted(names)
# group_tags = list(paths.group_tags_of_names(names))

# print("{} names, {} groups".format(len(names), len(group_tags)))

# pprint(group_tags)
# pprint(tags)
# np.random.shuffle(tags)

# x = [paths.img_of_name(fname) for fname in train_names]
# y = [paths.mask_of_name(fname) for fname in train_names]

# x = np.stack(x)
# y = np.stack(y)
# print('xy shapes   ', x.shape, y.shape)
# print("Input data: xshape:{}, xsize:{}GB, yshape:{}, ymean{:%}".format(
#     x.shape, x.size / 1024 ** 3,
#     y.shape, y.mean(),
# ))


# x = x.astype('uint8')
# y = y.astype(bool)

# exit()

import keras
import tensorflow as tf
import model


c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)


model_path = paths.get_latest_model_path_opt()

if model_path is None:
    print('Creating model')
    m = model._create_model()
    print('Creating test_names')
    test_names = paths.create_test_names()
    print('Saving infos')
    yaml.dump(dict(
        test_names=test_names,
    ), open(info_path, 'w'))

else:
    print('Loading model:', model_path)
    m = keras.models.load_model(model_path)
    print('Loading infos')
    info = yaml.load(open(info_path, 'r'))
    test_names = info['test_names']

m.summary()
train_names = [name for name in paths.create_names_list() if name not in test_names]
train_names = sorted(train_names)
print("{} train_names, {} test_names".format(len(train_names), len(test_names)))
pprint(sorted(test_names))

train_names = train_names


from prio_thread_pool import PrioThreadPool



# imgs = []
# masks = []
rasters = []

def work(name):
    global rasters
    rasters.append(
        (paths.img_of_name(name), paths.mask_of_name(name))
    )

PrioThreadPool(-1).iter(0, work, train_names)

x = [tup[0] for tup in rasters]
y = [tup[1] for tup in rasters]
del rasters

x = np.stack(x)
y = np.stack(y)
print('xy shapes   ', x.shape, y.shape)
print("Input data: xshape:{}, xsize:{}GB, yshape:{}, ymean{:%}".format(
    x.shape, x.size / 1024 ** 3,
    y.shape, y.mean(),
))

x = x.astype('uint8')
y = y.astype(bool)



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
# exit()
cbs = [ModelCheckpoint()]
m.fit(x, y, epochs=10000, batch_size=1, verbose=1, callbacks=cbs)
