import skimage.morphology
import skimage.filters
import skimage.measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from prio_thread_pool import PrioThreadPool
from constants import *

class Heatmap(object):

    def __init__(self, hm):
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

        hm = hm >= skimage.filters.threshold_yen(hm, 256)
        self.images.append((hm, 'thresholded (yen algo)'))

        kernel_size = max(1, np.rint(6 * WIDTH_RATIO_ORIGIN).astype(int)) # hyperparameter(s)
        hm = skimage.morphology.binary_closing(hm, skimage.morphology.disk(kernel_size))

        border = int(141 * WIDTH_RATIO_ORIGIN) # hyperparameter(s)
        mask = np.pad(np.ones(hm.shape - np.int_(border * 2)), border, 'constant', constant_values=0)
        hm = hm & mask.astype(bool)

        def _prop_ok(prop):
            # print("label {:03d}: area:{}".format(prop.label, prop.area))
            if not 211 * AREA_RATIO_ORIGIN < prop.area < 4746 * AREA_RATIO_ORIGIN: # hyperparameter(s)
                return False
            ma, mi = prop.major_axis_length, prop.minor_axis_length
            ratio = ma / mi
            # print("  ** ratio:{}".format(ratio))
            # if not 1.5 < ratio < 3.9: # hyperparameter(s)
            if not 1.5 < ratio < 4.05: # hyperparameter(s)
                return False
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
        self.centroids_yx = [
            tuple(np.asarray(prop.centroid).astype(int).tolist())
            for prop in props
        ]

def is_accurate(y, hm):
    """Work on 1 prediction

    Steps
    -----
    - Smooth truth
    - Convert truth to a its bounding rectangle
    - Build a list of centroids from predictions
    - Reject if more than 3 predictions
    - Find a prediction intersecting with truth
    """
    kernel_size = max(1, np.rint(7 * WIDTH_RATIO_ORIGIN))
    y = skimage.morphology.binary_closing(y, skimage.morphology.disk(kernel_size))
    yprops = skimage.measure.regionprops(
        ndimage.label(y, np.ones((3, 3)))[0]
    )
    if len(yprops) != 1:
        print('Warning, y {}/{} has {} objects'.format(
            i, len(ys), len(yprops)
        ))
        return False

    min_row, min_col, max_row, max_col = yprops[0].bbox
    y[min_row:max_row, min_col:max_col] = True

    hm = Heatmap(hm)
    if len(hm.centroids_yx) > 3:
        return False
    for cen in hm.centroids_yx:
        if y[cen]:
            return True
    return False

def accuracy(truths, preds):
    assert truths.shape == preds.shape
    success = 0

    def work(i):
        nonlocal success
        truth = truths[i]
        pred = preds[i]
        if is_accurate(truth, pred):
            success += 1

    PrioThreadPool(-1).iter(0, work, range(len(truths)))
    # for i in range(len(truths)):
        # work(i)
    return success / len(truths)
