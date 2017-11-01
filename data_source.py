import functools
import os

import numpy as np
from scipy import ndimage

from prio_thread_pool import PrioThreadPool
from constants import *

class DataSource(object):
    def __init__(self, prefix, dirname_img='img', dirname_mask='mask'):

        self.path_img = os.path.join(prefix, dirname_img)
        self.path_mask = os.path.join(prefix, dirname_mask)

        for tname in TEST_NAMES:
            assert tname in self.names

        self.names_train = [name for name in self.names if name not in TEST_NAMES]
        self.names_test = TEST_NAMES

    @property
    @functools.lru_cache(1)
    def names(self):
        print('Computing names for the first time...')
        set1 = {
            fname[:-4]
            for fname in os.listdir(self.path_img)
            if fname.endswith('.png')
        }
        set2 = {
            fname[:-4]
            for fname in os.listdir(self.path_mask)
            if fname.endswith('.png')
        }
        return sorted(list(set1 & set2))

    @property
    @functools.lru_cache(1)
    def xtrain(self):
        print('Computing xtrain for the first time...')

        count = len(self.names_train)
        imgs = [None] * count

        def _work(i):
            nonlocal imgs
            name = self.names_train[i]
            imgs[i] = self.img_of_name(name)

        PrioThreadPool(-1).iter(0, _work, range(count))
        return np.stack(imgs)

    @property
    @functools.lru_cache(1)
    def ytrain(self):
        print('Computing ytrain for the first time...')
        return np.stack([
            self.mask_of_name(name) for name in self.names_train
        ])

    @property
    @functools.lru_cache(1)
    def xtest(self):
        print('Computing xtest for the first time...')
        count = len(self.names_test)
        imgs = [None] * count

        def _work(i):
            nonlocal imgs
            name = self.names_test[i]
            imgs[i] = self.img_of_name(name)

        PrioThreadPool(-1).iter(0, _work, range(count))
        return np.stack(imgs)

    @property
    @functools.lru_cache(1)
    def ytest(self):
        print('Computing xtest for the first time...')
        return np.stack([
            self.mask_of_name(name) for name in self.names_test
        ])

    @functools.lru_cache(None)
    def img_of_name(self, name):
        shapes = []

        path = os.path.join(self.path_img, name + '.png')
        arr = ndimage.imread(path).astype('uint8')
        shapes.append(str(arr.shape))

        arr = ndimage.zoom(arr, (img_h / 1080, img_w / 1920, 1))
        shapes.append(str(arr.shape))

        print("Read img  {}, shapes:{}".format(
            name, ' -> '.join(shapes)
        ))
        return arr

    @functools.lru_cache(None)
    def mask_of_name(self, name):
        shapes = []

        path = os.path.join(self.path_mask, name + '.png')
        arr = ndimage.imread(path)
        shapes.append(str(arr.shape))

        arr = ndimage.zoom(arr, (img_h / 1080, img_w / 1920), order=1)
        shapes.append(str(arr.shape))

        arr = arr.astype('bool')

        print("Read mask {}, shapes:{}".format(
            name, ' -> '.join(shapes)
        ))
        return arr
