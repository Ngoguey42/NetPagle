

import buzzard as buzz
import matplotlib.pyplot as plt
import numpy as np
import skimage
# import skimage.morphology
import scipy.ndimage as ndi
import shapely.geometry as sg
import descartes

class ModelShow(object):

    def _show_pretty_cruve(self, ax, x, y, color, label):
        def _steps():
            i = 0
            start = 0
            step = 3
            while start <= border_dist.max():
                yield i, start, start + step
                start += step
                step += 1
                i += 1

        def _kernel(radius, strength):
            a = np.zeros(radius * 2 + 1, float)
            a[radius] = 1
            a = ndi.gaussian_filter1d(a, radius / (4 / strength))
            return a

        def _smooth(y, strength):
            res = np.zeros(y.shape, dtype=float)
            for radius, start, end in _steps():
                kernel = _kernel(radius, strength)
                res += (
                    ndi.filters.convolve1d(y, kernel, mode='nearest') *
                    ((border_dist >= start) & (border_dist < end))
                )
            return res

        border_dist = np.min(
            np.c_[x, x.max() - x],
            axis=1,
        )
        ysmooth = _smooth(y, 2)

        xy = np.c_[x, y]


        ax.plot(
            xy[:, 0], xy[:, 1], '.',
            ms=1/2, alpha=1, c=color, zorder=40, aa=True,
        )
        ax.plot(
            xy[:, 0], ysmooth,
            lw=1, alpha=1, ls='-', c=color, zorder=50,
            label='{} (max={})'.format(label, y.max()),
        )

        # fp = buzz.Footprint(
        #     tl=(xy[:, 0].min(), xy[:, 1].max()),
        #     size=(xy[:, 0].ptp(), xy[:, 1].ptp()),
        #     rsize=(1920 // 2, 1080 // 2),
        # ).dilate(50)

        # # tlx, dx, rx, tly, ry, dy = fp.gt

        # with buzz.Env(allow_complex_footprint=True):
        #     fp2 = buzz.Footprint(
        #         gt=(fp.rtlx, 1, 0, fp.rtly, 0, 1),
        #         rsize=fp.rsize,
        #     )

        # mask_points = np.zeros(fp.shape, bool)
        # for pt in xy:
        #     x, y = fp.spatial_to_raster(pt)
        #     mask_points[y, x] = 1
        # mask_points |= fp.burn_lines(sg.LineString(np.c_[xy[:, 0], ysmooth]))

        # dist = ndi.morphology.distance_transform_edt(~mask_points)

        # abovexy = np.c_[xy[:, 0], np.maximum.accumulate(xy[:, 1])]
        # belowxy = np.c_[xy[:, 0], np.minimum.accumulate(xy[::-1, 1])[::-1]]
        ax.fill_between(
            xy[:, 0],
            np.maximum.accumulate(xy[:, 1]),
            np.minimum.accumulate(xy[::-1, 1])[::-1],
            facecolor=color,
            alpha=1/3,
            zorder=10,
        )

        # polyxy = np.vstack([abovexy, belowxy[::-1]])
        # poly = sg.Polygon(polyxy)
        # mask_poly = fp.burn_polygons(poly)
        # if mask_poly.sum() == 0:
        #     return

        # for max_dist in np.arange(dist[mask_poly].max(), 40, -1):
        #     erodable = (dist >= max_dist) & mask_poly
        #     eroded = skimage.morphology.binary_erosion(mask_poly, skimage.morphology.disk(1)) ^ mask_poly
        #     eroded = eroded & erodable
        #     mask_poly = mask_poly & ~eroded

        # mask_poly = skimage.morphology.binary_closing(mask_poly, skimage.morphology.disk(5))

        # for poly in fp2.find_polygons(mask_poly):
        #     if poly.area < 100:
        #         continue
        #     print('poly')
        #     print('  ', poly.area)
        #     poly = poly.simplify(2 ** 0.5, False)
        #     print('  ', poly.area)
        #     poly = sg.Polygon([
        #         fp.raster_to_spatial(pt)
        #         for pt in np.asarray(poly.exterior)
        #     ])
        #     ax.add_patch(
        #         descartes.PolygonPatch(poly, zorder=10, alpha=1/3, fc=color, ec='white', linewidth=0),
        #     )

    def show_board(self):

        if self.df.shape[0] < 4:
            print('No plot when less than 4 elements!')
            return

        fig, ax1 = plt.subplots(figsize=(16, 9))

        self._show_pretty_cruve(ax1, self.df.epoch, self.df.acctrain, 'red', 'Accuracy train set')
        self._show_pretty_cruve(ax1, self.df.epoch, self.df.acctest, 'green', 'Accuracy test set')
        ax1.plot(
            self.df.epoch, self.df.lr,
            lw=2, alpha=1/3, ls='-', c='orange', label='Learning rate', zorder=20
        )
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('percent')
        ax1.set_title(self.directory)
        plt.legend(loc='lower center')

        ax2 = ax1.twinx()
        mask = self.df.loss < self.df.loss.min() + self.df.loss.std() * 3
        ax2.plot(
            self.df.epoch[mask],
            self.df.loss[mask],
            lw=1, alpha=1, ls='-', c='purple',
            label='loss (min={:.8f})'.format(self.df.loss.min()),
            zorder=45,
        )
        ax2.set_ylabel('loss', color='purple')
        ax2.tick_params('y', colors='purple')
        plt.legend(loc='lower left')

        plt.tight_layout()

        plt.savefig(
            self.directory + '/status.png',
            dpi=200, orientation='landscape', bbox_inches='tight',
        )
        plt.close('all')
