print('Snap!')
import mss
import numpy as np

with mss.mss() as sct:
    monitor = sct.monitors[1]
    i = sct.grab(monitor)
    print('Snaped!')
    i = np.asarray(i)[:, :, [2, 1, 0]]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import misc
import os
import sys
import pytz
import datetime
import names
import skimage.segmentation
import string

import data_source
from constants import *

class Segmentozor(object):
    def __init__(self, img):
        self._input = np.full(img.shape[:2], 0, np.int8)
        self._count = 0
        self._img = img
        self.dirty = False

    def remove_at(self, x, y):
        x = int(x)
        y = int(y)
        if self._input[y, x] != 0:
            if self._count >= 3:
                self.dirty = True
            self._count -= 1
            self._input[y, x] = 0

    def add_fg(self, x, y):
        x = int(x)
        y = int(y)
        if self._input[y, x] != 1:
            if self._count >= 2:
                self.dirty = True
            self._count += 1
            self._input[y, x] = 1

    def add_bg(self, x, y):
        x = int(x)
        y = int(y)
        if self._input[y, x] != 2:
            if self._count >= 2:
                self.dirty = True
            self._count += 1
            self._input[y, x] = 2

    def build_image(self):
        self.dirty = False
        if self._count < 3:
            return np.full(self._input.shape, -1, np.uint8)
        a = self._input.copy()
        hull = skimage.morphology.convex_hull_image(
            self._input.astype(bool)
        )
        a[:] = -1
        a[hull] = 0
        a[self._input == 1] = 1
        a[self._input == 2] = 2
        seg = skimage.segmentation.random_walker(
            self._img, a, multichannel=True,
        )
        return seg


class DraggableRectangle:
    def __init__(self, rect, seg, aximgs):
        self.rect = rect
        self.press = None
        self._redraw = False
        self.seg = seg
        self._pressed = []
        self.aximgs = aximgs

    def connect(self):
        'connect to all the events we need'
        print('connect')
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidlole = self.rect.figure.canvas.mpl_connect(
            'key_press_event', self.key_press_callback)
        # self.cidmotion = self.rect.figure.canvas.mpl_connect(
            # 'motion_notify_event', self.on_motion)
        self.tamerlap = self.rect.figure.canvas.mpl_connect(
            'draw_event', self.draw_callback
        )
        self.plus = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.motion_notify_callback
        )

    def draw_callback(self, event):
        if self._redraw:
            labels = self.seg.build_image()
            imgmask = labels == 1
            img1 = self.seg._img.copy()
            img1[imgmask] = (img1[imgmask] * 1.8).clip(0, 255).astype(np.uint8)
            img1[~imgmask] = (img1[~imgmask] / 1.8).clip(0, 255).astype(np.uint8)

            img2 = self.seg._img.copy()
            img2[self.seg._input == 1] = [255, 0, 0]
            img2[self.seg._input == 2] = [0, 255, 0]

            self.aximgs[0].set_data(img1)
            self.aximgs[1].set_data(img2)

            self._redraw = False
            self.rect.figure.canvas.draw()

    def key_press_callback(self, event):
        if event.key=='e':
            self._redraw = True
            self.rect.figure.canvas.draw()

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if plt.get_current_fig_manager().toolbar.mode != '':
            return
        if event.xdata is None:
            return
        if event.button in {1, 2, 3}:
            self._pressed += [event.button]
            x, y = event.xdata + 0.5, event.ydata + 0.5
            if event.button == 1:
                self.seg.add_fg(x, y)
            elif event.button == 3:
                self.seg.add_bg(x, y)
            elif event.button == 2:
                self.seg.remove_at(x, y)

    def on_release(self, event):
        'on release we reset the press data'
        if event.button in {1, 2, 3}:
            self._pressed = [
                k
                for k in self._pressed
                if k != event.button
            ]
        # self.rect.figure.canvas.draw()

    def motion_notify_callback(self, event):
        if event.xdata is None:
            return
        for k in self._pressed[::-1]:
            x, y = event.xdata + 0.5, event.ydata + 0.5
            if k == 1:
                self.seg.add_fg(x, y)
            elif k == 3:
                self.seg.add_bg(x, y)
            elif k == 2:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        self.seg.remove_at(x + i, y + j)
            break

    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

def mask_of_img(img, name):
    seg = Segmentozor(img)
    fig = plt.figure(figsize=(16, 16 / 16 * 9))

    axes = []
    aximgs = []
    imgs = [img, img]
    tags = ['segmentation', 'user inputs']

    for i, img in enumerate(imgs):
        if i == 0:
            a = fig.add_subplot(1, 2, i + 1)
            a.set_title(name)
        else:
            a = fig.add_subplot(1, 2, i + 1, sharex=axes[0], sharey=axes[0])
        axes.append(a)
        ai = plt.imshow(img)
        aximgs.append(ai)

    rects = axes[0].bar(range(1), 20*np.random.rand(1))
    drs = []
    for rect in rects:
        dr = DraggableRectangle(rect, seg, aximgs)
        dr.connect()
        drs.append(dr)

    plt.show()
    labels = seg.build_image()
    return labels == 1

img = np.asarray(i)
print('img', img.shape)
for i in range(3):
    print("  channel {}: mean:{}".format(i, img[..., i].mean()))

ds = data_source.DataSource(PREFIX)
tags = [
    ''.join(
        c
        for c in s.lower()
        if c in string.ascii_lowercase + '-' + '0123456789'
    )
    for s in sys.argv[1:]
]

name = ds.create_name(tags)
mask = mask_of_img(img, name)
print("mask: count:{}, mean:{:%}".format(mask.sum(), mask.mean()))
ds.save_new_data(img, mask, name)
