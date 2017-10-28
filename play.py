import yaml

import numpy as np
import matplotlib.pyplot as plt
# from keras import models
import keras
import tensorflow as tf

import paths
from constants import *

def show_prediction_of_name(name):
    img = paths.img_of_name(name)
    print('predicting', name)
    pred = m.predict(img[np.newaxis, ...])
    print(pred.shape)
    pred = pred[0]
    print(f'pred min{pred.min()} max{pred.max()} mean{pred.mean()}')
    # perc99 = np.percentile(pred, 99)
    # pred = pred.clip(0, perc99)
    # print(f'pred min{pred.min()} max{pred.max()} mean{pred.mean()}')
    # pred = 1 - pred
    # print(f'pred min{pred.min()} max{pred.max()} mean{pred.mean()}')
    pred = (pred - pred.min()) / pred.ptp()
    print(f'pred min{pred.min()} max{pred.max()} mean{pred.mean()}')
    print("showing {}'s heatmap".format(name))
    fig = plt.figure(figsize=(13, 13 / 16 * 9))
    mask = paths.mask_of_name(name).reshape(img_h, img_w)
    # mask = _mask_of_name(name)[..., 0]
    # img = np.moveaxis(img, 0, 2)
    img2 = img.copy()
    img2[mask] = (img2[mask] * 1.8).clip(0, 255).astype('uint8')
    img2[~mask] = (img2[~mask] / 1.8).clip(0, 255).astype('uint8')
    imgs = [pred, mask, img, img2]
    tags = ['heat map', 'input Y', 'input X', 'input X and Y']
    a1 = None
    for i, (arr, title) in enumerate(zip(imgs, tags)):
        if i == 0:
            a = fig.add_subplot(2, 2, i + 1)
            a1 = a
        else:
            a = fig.add_subplot(2, 2, i + 1, sharex=a1, sharey=a1)
        imgplot = plt.imshow(arr)
        a.set_title(title)
    plt.show()


c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)

model_path = paths.get_latest_model_path_opt()
assert model_path is not None
print('Loading model:', model_path)
m = keras.models.load_model(model_path)


names = paths.create_names_list()
test_names = yaml.load(open(info_path, 'r'))['test_names']
train_names = [name for name in paths.create_names_list() if name not in test_names]

# for name in names:
# for name in train_names:
for name in test_names:
    # if 'honor' not in name:
    #     continue
    show_prediction_of_name(name)
