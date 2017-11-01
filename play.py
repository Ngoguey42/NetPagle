import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import descartes
import shapely.geometry as sg

import paths
from constants import *
import accuracy
import data_source
import model

def show_many_images(imgs, tags, patchess):
    count = len(imgs)
    assert len(imgs) == len(tags) == len(patchess)

    if count == 1: w, h = 1, 1
    elif count == 2: w, h = 2, 1
    elif count in {3, 4}: w, h = 2, 2
    elif count in {5, 6}: w, h = 3, 2
    elif count in {7, 8, 9}: w, h = 3, 3
    else: assert False

    fig = plt.figure(figsize=(13, 13 / 16 * 9))
    for i, img, title, patches in zip(range(count), imgs, tags, patchess):
        if i == 0:
            a = fig.add_subplot(h, w, i + 1)
            a0 = a
        else:
            a = fig.add_subplot(h, w, i + 1, sharex=a0, sharey=a0)
        plt.imshow(img)
        a.set_title(title)
        for patch in patches:
            a.add_patch(patch)
    plt.show()

def show_prediction(name, x, y):
    print("Predicting {}".format(name), x.shape, y.shape)
    hm = m.m.predict(np.asarray([x]), 1, True)[0]
    print(f'heat-map min{hm.min()} max{hm.max()} mean{hm.mean()}')

    hm = accuracy.Heatmap(hm)
    centroids = [np.flipud(cen) for cen in hm.centroids_yx]
    print(f'{len(centroids)} centroids -> {centroids}')


    imgs = [y, x] + [img for (img, _) in hm.images]
    patchess = [()] * len(imgs)
    patchess[-1] = (
        # [descartes.PolygonPatch(sg.Point(cen).buffer(2), alpha=1., zorder=2, fc='white')
        #  for cen in centroids] +
        [descartes.PolygonPatch(sg.Point(cen).buffer(15), alpha=0.4, zorder=1, fc='white', ec='red', lw=2)
         for cen in centroids]
    )
    tags = ['x', 'y'] + [title for (_, title) in hm.images]
    show_many_images(
        imgs, tags, patchess,
    )

c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)

ds = data_source.DataSource(PREFIX)
m = model.Model(os.path.join(PREFIX, sys.argv[1]), ds)

print(m.eval_accuracies())

# exit()
# for name, x, y, i in zip(ds.names, ds.xtrain, ds.ytrain, range(100000)):
for i, name in enumerate(TEST_NAMES):
# for i, name in enumerate([
# 	'17-10-24-23-10-39_blue-thunderbluff-courtyard-scroll0_marilyn',
# 	'17-10-24-23-28-52_blue-darnassus-auctionhouse-scroll0_gina',
# 	'17-10-24-23-36-13_blue-darnassus-temple-scroll0_kelly-occlusion',
# 	'17-10-28-21-24-48_red-stonetalon-sunrock-scroll10_anna',
# 	'17-10-27-00-33-08_green-moonglade-river-scroll0_lorraine',
# 	'17-10-28-19-55-15_black-silverpine-lake-scroll5_norma-occlusion',
# 	'17-10-28-20-08-44_black-silverpine-bridge-scroll5_steven',
# ]):
    # if not ('gina' in name or 'raym' in name):
    #     continue

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(i, name)
    x = ds.img_of_name(name)
    y = ds.mask_of_name(name)
    show_prediction(name, x, y)
# for name, x, y, i in zip(ds.names, ds.xtest, ds.ytest, range(100000)):
#     print(i)
#     show_prediction(name, x, y)
    # if i > 2:
        # break
