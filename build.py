import PIL.ImageGrab as pig

print('Snap!')
i = pig.grab()
print('Snaped!')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
# from pylab import *
from scipy import misc
import os
import pytz
import datetime
import names
import skimage.segmentation

import paths


# def mask_of_img(img):

#     points = None

#     def onselect(eclick, erelease):
#         nonlocal points

#         points = np.rint((
#             (eclick.xdata, eclick.ydata),
#             (erelease.xdata, erelease.ydata),
#         )).astype(int)
#         print(points)

#     height = 18
#     fig, ax = plt.subplots(figsize=(height, height / 16 * 9))
#     plt.imshow(img)

#     rs = RectangleSelector(ax, onselect, drawtype='box')
#     plt.show()
#     assert points is not None

#     startx = points[:, 0].min()
#     endx = points[:, 0].max() + 1
#     starty = points[:, 1].min()
#     endy = points[:, 1].max() + 1

#     mask = np.zeros(img.shape[:2], bool)
#     mask[starty:endy, startx:endx] = True
#     return mask


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
            print('remove_at', x, y)
            if self._count >= 3:
                self.dirty = True
            self._count -= 1
            self._input[y, x] = 0

    def add_fg(self, x, y):
        x = int(x)
        y = int(y)
        if self._input[y, x] != 1:
            print('add_fg', x, y)
            if self._count >= 2:
                self.dirty = True
            self._count += 1
            self._input[y, x] = 1

    def add_bg(self, x, y):
        x = int(x)
        y = int(y)
        if self._input[y, x] != 2:
            print('add_bg', x, y)
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
    def __init__(self, rect, seg):
        self.rect = rect
        self.press = None
        self._redraw = False
        self.seg = seg

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

    def draw_callback(self, event):
        print('draw_callback', self._redraw)
        if self._redraw:
            labels = self.seg.build_image()
            imgmask = labels == 1
            img = self.seg._img.copy()
            img[imgmask] = (img[imgmask] * 1.8).clip(0, 255).astype(np.uint8)
            img[~imgmask] = (img[~imgmask] / 1.8).clip(0, 255).astype(np.uint8)
            img[self.seg._input == 1] = [255, 0, 0]
            img[self.seg._input == 2] = [0, 255, 0]
            plt.imshow(img)
            assert not self.seg.dirty
            self._redraw = False
            self.rect.figure.canvas.draw()

    def key_press_callback(self, event):
        if event.key=='e':
            print('key_press_callback')
            self._redraw = True
            self.rect.figure.canvas.draw()

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if plt.get_current_fig_manager().toolbar.mode != '':
            return

        x, y = event.xdata, event.ydata
        print('on_press', x, y, event.button)
        print(event)
        print()

        print(dir(self))

        if event.button == 1:
            self.seg.add_fg(x, y)
        elif event.button == 3:
            self.seg.add_bg(x, y)
        elif event.button == 2:
            self.seg.remove_at(x, y)
        # if event.inaxes != self.rect.axes: return

        # contains, attrd = self.rect.contains(event)
        # if not contains: return
        # print('event contains', self.rect.xy)
        # x0, y0 = self.rect.xy
        # self.press = x0, y0, event.xdata, event.ydata

    def on_release(self, event):
        'on release we reset the press data'
        print('on_release')
        self.press = None
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

def mask_of_img(img):
    seg = Segmentozor(img)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.imshow(img)
    rects = ax.bar(range(1), 20*np.random.rand(1))
    drs = []
    for rect in rects:
        dr = DraggableRectangle(rect, seg)
        dr.connect()
        drs.append(dr)

    plt.show()
    labels = seg.build_image()
    return labels == 1

img = np.asarray(i)
print('img', img.shape)
mask = mask_of_img(img)
print('mask {!r} {:%}'.format(mask.shape, mask.mean()))

name = paths.create_name()
print(name)
path_img = paths.img_path_of_name(name)
path_mask = paths.mask_path_of_name(name)

misc.imsave(path_img, img)
misc.imsave(path_mask, mask)
