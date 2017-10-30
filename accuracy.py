import skimage.morphology
import skimage.filters
import skimage.measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from prio_thread_pool import PrioThreadPool
from constants import *

def centroids_of_heatmap(hm):
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
    hm = hm >= skimage.filters.threshold_otsu(hm, 256)

    kernel_size = max(1, np.rint(6 * WIDTH_RATIO_ORIGIN)) # hyperparameter(s)
    hm = skimage.morphology.binary_closing(hm, skimage.morphology.disk(kernel_size))

    border = int(141 * WIDTH_RATIO_ORIGIN) # hyperparameter(s)
    mask = np.pad(np.ones(hm.shape - np.int_(border * 2)), border, 'constant', constant_values=0)
    hm = hm & mask.astype(bool)

    def _prop_ok(prop):
        if not 211 * AREA_RATIO_ORIGIN < prop.area < 4746 * AREA_RATIO_ORIGIN: # hyperparameter(s)
            return False
        ma, mi = prop.major_axis_length, prop.minor_axis_length
        ratio = ma / mi
        # print("label {:03d}: area:{}, ratio:{}".format(prop.label, prop.area, ratio))
        if not 1.5 < ratio < 3.5: # hyperparameter(s)
            return False
        return True

    lbl, nlbl = ndimage.label(hm)
    props = skimage.measure.regionprops(lbl)
    props = [prop for prop in props if _prop_ok(prop)]

    # if len(props) == 0:
    #     lbls = np.arange(1, nlbl+1)
    #     hm = np.isin(lbl, [prop.label for prop in props])
    #     plt.imshow(lbl); plt.show()
    #     plt.imshow(hm); plt.show()
    return [prop.centroid for prop in props]

def accuracy(truths, preds):
    assert truths.shape == preds.shape
    success = 0

    def work(i):
        """Work on 1 prediction

        Steps
        -----
        - Smooth truth
        - Convert truth to a its bounding rectangle
        - Build a list of centroids from predictions
        - Reject if more than 3 predictions
        - Find a prediction intersecting with truth
        """
        nonlocal success

        truth = truths[i]
        pred = preds[i]

        kernel_size = max(1, np.rint(7 * WIDTH_RATIO_ORIGIN))
        truth = skimage.morphology.binary_closing(truth, skimage.morphology.disk(kernel_size))
        truthprops = skimage.measure.regionprops(
            ndimage.label(truth, np.ones((3, 3)))[0]
        )
        if len(truthprops) != 1:
            print('Warning, truth {}/{} has {} objects'.format(
                i, len(truths), len(truthprops)
            ))
            return

        min_row, min_col, max_row, max_col = truthprops[0].bbox
        truth[min_row:max_row, min_col:max_col] = True

        centroids = centroids_of_heatmap(pred)
        if len(centroids) > 3:
            return
        for centroid in centroids:
            centroid = tuple(np.rint(centroid).astype(int))
            if truth[centroid]:
                success += 1
                break

    # PrioThreadPool(-1).iter(0, work, range(len(truths)))
    for i in range(len(truths)):
        work(i)
    return success / len(truths)
