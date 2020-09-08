import skimage.morphology
import skimage.filters
import skimage.measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from prio_thread_pool import PrioThreadPool
from constants import *

_print = print

class Heatmap(object):

    def __init__(self, hm, verbose=False):
        """
        Steps
        -----
        - Threshold heatmap
        - Close objects
        - Ignore outer border
        - Filter area
        - Filter ellipse major/ellipse minor

        Returns
        -------
        list of centroids
        """
        self.hm = hm
        self.images = [(hm, 'heatmap')]

        if not verbose:
            def _f(*args): pass
            print = _f
        else:
            print = _print

        hm = hm >= skimage.filters.threshold_yen(hm, 256)
        self.images.append((hm, 'thresholded (yen algo)'))

        kernel_size = max(1, np.rint(6 * WIDTH_RATIO_ORIGIN).astype(int)) # hyperparameter(s)
        hm = skimage.morphology.binary_closing(hm, skimage.morphology.disk(kernel_size))

        border = int(141 * WIDTH_RATIO_ORIGIN) # hyperparameter(s)
        mask = np.pad(np.ones(hm.shape - np.int_(border * 2)), border, 'constant', constant_values=0)
        hm = hm & mask.astype(bool)

        def _prop_ok(prop):
            print("label {:03d}: area:{}".format(prop.label, prop.area))
            if not 211 * AREA_RATIO_ORIGIN < prop.area < 4746 * AREA_RATIO_ORIGIN: # hyperparameter(s)
                return False
            # ma, mi = prop.major_axis_length, prop.minor_axis_length
            # ratio = ma / mi
            # print("  ** ratio:{}".format(ratio))
            # if not 1.5 < ratio < 4.05: # hyperparameter(s)
                # return False
            return True

        lbl, nlbl = ndimage.label(hm, np.ones((3, 3)))
        self.images.append((lbl, 'close radius {}px, remove border {}px'.format(
            kernel_size, border,
        )))

        props = skimage.measure.regionprops(lbl)
        props = [prop for prop in props if _prop_ok(prop)]
        lbl = np.isin(lbl, [prop.label for prop in props])
        self.images.append((lbl, 'filter area and ellipse radiuses'.format()))

        kernel_size = max(1, np.rint(43 * WIDTH_RATIO_ORIGIN).astype(int)) # hyperparameter(s)
        mask = skimage.morphology.binary_closing(lbl != 0, skimage.morphology.disk(kernel_size))
        lbl, nlbl = ndimage.label(mask, np.ones((3, 3)))

        self.images.append((lbl, 'close of {}px'.format(
            kernel_size,
        )))

        props = skimage.measure.regionprops(lbl)
        yx_of_prop = lambda prop: tuple(np.asarray(prop.centroid).astype(int).tolist())
        props = sorted(props, key=lambda p: p.area, reverse=True)
        self.centroids_yx = [yx_of_prop(prop) for prop in props]

def analyse_prediction(y, hm):
    """Work on 1 prediction

    Steps
    -----
    - Smooth truth
    - Convert truth to a its bounding rectangle
    """
    kernel_size = max(1, np.rint(7 * WIDTH_RATIO_ORIGIN))
    y = skimage.morphology.binary_closing(y, skimage.morphology.disk(kernel_size))
    yprops = skimage.measure.regionprops(
        ndimage.label(y, np.ones((3, 3)))[0]
    )
    target_count = len(yprops)
    if len(yprops) not in {0, 1}:
        print('Warning, y has {} targets'.format(
            target_count,
        ))
    if target_count == 0:
        return 0, 0

    min_row, min_col, max_row, max_col = yprops[0].bbox
    y[min_row:max_row, min_col:max_col] = True

    hm = Heatmap(hm)
    hit_count = 0
    for _, centroid_yx in zip(range(target_count), hm.centroids_yx):
        if y[centroid_yx]:
            hit_count += 1
    return target_count, hit_count

def accuracy(truths, preds):
    assert truths.shape == preds.shape
    targets = 0
    hits = 0

    def work(i):
        nonlocal targets, hits
        truth = truths[i]
        pred = preds[i]
        t, h = analyse_prediction(truth, pred)
        targets += t
        hits += h

    PrioThreadPool(-1).iter(0, work, range(len(truths)))
    print("Accuracy of {}/{} = {:%} on {} images".format(
        hits, targets, hits / targets, len(truths),
    ))
    return hits / targets
