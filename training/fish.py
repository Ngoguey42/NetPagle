import time
import sys
import os

import keras
import tensorflow as tf
import pyautogui
import mss
import numpy as np

import model
import data_source
from constants import *
import accuracy

c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
keras.backend.tensorflow_backend.set_session(sess)

ds = data_source.DataSource(PREFIX)
m = model.Model(os.path.join(PREFIX, sys.argv[1]), ds)

def run():
    # print('Press enter to launch')
    # sys.stdin.readline()

    pyautogui.click(500, 500)
    pyautogui.moveTo(1893, 950)
    pyautogui.click(clicks=1, duration=0.01, interval=0.2)
    time.sleep(2.5)
    print('ok')
    print('Snap!')

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        print('Snaped!')
    img = np.asarray(img)[:, :, [2, 1, 0]]
    img = ds.zoom(img)
    hm = m.m.predict(np.asarray([img]), 1, False)[0]
    hm = accuracy.Heatmap(hm)

    if not hm.centroids_yx:
        print('No centroids')
    else:
        coord = np.flipud(hm.centroids_yx[0])
        coord = coord / WIDTH_RATIO_ORIGIN
        print('Centroid at', coord)
        pyautogui.moveTo(*coord)
    time.sleep(2.5)
    pyautogui.press('space')
    time.sleep(1)

while True: run()
