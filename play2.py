

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

path1 = 'C:/Users/Ngo/Desktop/fishdb/test.jpg'
path2 = 'C:/Users/Ngo/Desktop/fishdb/testlab.png'

img = misc.imread(path1).astype('uint8')[..., :3].astype('uint8')
mask = misc.imread(path2).astype('uint8')[..., :3]

inside = np.all(mask == [0, 255, 0], axis=-1)
outside = np.all(mask == [255, 0, 0], axis=-1)
mask = outside | inside

print("inside {:%}".format(inside.mean()))
print("outside {:%}".format(outside.mean()))
print("mask {:%}".format(mask.mean()))

# plt.imshow(mask)
# plt.show()

# mask.any(axis=0)
vec = np.cumsum(mask.any(axis=1))
miny = np.arange(vec.size)[vec == 0].max()
maxy = np.arange(vec.size)[vec == vec[-1]].min()
vec = np.cumsum(mask.any(axis=0))
minx = np.arange(vec.size)[vec == 0].max()
maxx = np.arange(vec.size)[vec == vec[-1]].min()

print(miny, minx, maxy, maxx)

b = np.zeros((1,65),np.float64)
f = np.zeros((1,65),np.float64)

import skimage.segmentation

# mask = np.zeros(img.shape[:2], 'int8')
# mask[:] = -1
# mask[miny:maxy + 1, minx:maxx + 1] = 0
# mask[inside] = 1
# mask[outside] = 2
# mask = skimage.segmentation.random_walker(
#     img,
#     mask,
#     multichannel=True,
# )
# imgmask = mask == 1

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


seg = Segmentozor(img)

import numpy as np
import matplotlib.pyplot as plt

class DraggableRectangle:
    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self._redraw = False

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
            labels = seg.build_image()
            imgmask = labels == 1
            img = seg._img.copy()
            img[imgmask] = (img[imgmask] * 1.8).clip(0, 255).astype(np.uint8)
            img[~imgmask] = (img[~imgmask] / 1.8).clip(0, 255).astype(np.uint8)
            img[seg._input == 1] = [255, 0, 0]
            img[seg._input == 2] = [0, 255, 0]
            plt.imshow(img)
            assert not seg.dirty
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
            seg.add_fg(x, y)
        elif event.button == 3:
            seg.add_bg(x, y)
        elif event.button == 2:
            seg.remove_at(x, y)
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

fig = plt.figure()
ax = fig.add_subplot(111)

plt.imshow(img)
rects = ax.bar(range(1), 20*np.random.rand(1))
drs = []
for rect in rects:
    dr = DraggableRectangle(rect)
    dr.connect()
    drs.append(dr)

plt.show()

# i = seg.build_image()
# print(np.stack(np.unique(i, return_counts=True)))

# seg.add_fg(50, 50)
# i = seg.build_image()
# print(np.stack(np.unique(i, return_counts=True)))

# seg.add_fg(55, 55)
# i = seg.build_image()
# print(np.stack(np.unique(i, return_counts=True)))

# seg.add_fg(50, 55)
# i = seg.build_image()
# print(np.stack(np.unique(i, return_counts=True)))

# seg.add_fg(55, 50)
# i = seg.build_image()
# print(np.stack(np.unique(i, return_counts=True)))





# mask = np.zeros(img.shape[:2], np.uint8)
# mask[:] = cv2.GC_BGD
# mask[miny:maxy + 1, minx:maxx + 1] = cv2.GC_PR_FGD
# mask[inside] = cv2.GC_FGD
# mask[outside] = cv2.GC_BGD
# mask, bgdModel, fgdModel = cv2.grabCut(
#     img,
#     mask,
#     None,
#     b, f,
#     5, cv2.GC_INIT_WITH_RECT | cv2.GC_INIT_WITH_MASK,
# )
# imgmask = (mask == 3) | (mask == 1)



# img[imgmask] = (img[imgmask] * 1.8).clip(0, 255).astype(np.uint8)
# img[~imgmask] = (img[~imgmask] / 1.8).clip(0, 255).astype(np.uint8)

# plt.imshow(img)
# plt.show()


exit()



"""
Interactive tool to draw mask on an image or image-like array.
Adapted from matplotlib/examples/event_handling/poly_editor.py
"""
import numpy as np

# import matplotlib as mpl
# mpl.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.mlab import dist_point_to_segment
# from matplotlib import nxutils


class MaskCreator(object):
    """An interactive polygon editor.
    Parameters
    ----------
    poly_xy : list of (float, float)
        List of (x, y) coordinates used as vertices of the polygon.
    max_ds : float
        Max pixel distance to count as a vertex hit.
    Key-bindings
    ------------
    't' : toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
    'd' : delete the vertex under point
    'i' : insert a vertex at point.  You must be within max_ds of the
          line connecting two existing vertices
    """
    def __init__(self, ax, poly_xy=None, max_ds=10):
        self.showverts = True
        self.max_ds = max_ds
        if poly_xy is None:
            poly_xy = default_vertices(ax)
        self.poly = Polygon(poly_xy, animated=True,
                            fc='y', ec='none', alpha=0.4)

        ax.add_patch(self.poly)
        ax.set_clip_on(False)
        ax.set_title("Click and drag a point to move it; "
                     "'i' to insert; 'd' to delete.\n"
                     "Close figure when done.")
        self.ax = ax

        x, y = list(zip(*self.poly.xy))
        self.line = plt.Line2D(x, y, color='none', marker='o', mfc='r',
                               alpha=0.2, animated=True)
        self._update_line()
        self.ax.add_line(self.line)

        self.poly.add_callback(self.poly_changed)
        self._ind = None # the active vert

        canvas = self.poly.figure.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    # def get_mask(self, shape):
    #     """Return image mask given by mask creator"""
    #     h, w = shape
    #     y, x = np.mgrid[:h, :w]
    #     points = np.transpose((x.ravel(), y.ravel()))
    #     mask = nxutils.points_inside_poly(points, self.verts)
    #     return mask.reshape(h, w)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        #Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return
        self._ind = self.get_ind_under_cursor(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        ignore = not self.showverts or event.button != 1
        if ignore:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key=='d':
            ind = self.get_ind_under_cursor(event)
            if ind is None:
                return
            if ind == 0 or ind == self.last_vert_ind:
                print("Cannot delete root node")
                return
            self.poly.xy = [tup for i,tup in enumerate(self.poly.xy)
                                if i!=ind]
            self._update_line()
        elif event.key=='i':
            p = event.x, event.y # cursor coords
            pd = (event.xdata, event.ydata)

            count = len(self.poly.xy)
            print('event i:')
            print('  p', p)
            print('  pd', pd)
            print('  count', count)
            if count in {0, 1}:
                self.poly.xy.append(pd)
            elif count == 2:
                self.poly.xy.append(pd)
                self.poly.xy.append(self.poly.xy[0])
            else:
                xys = self.poly.get_transform().transform(self.poly.xy)
                for i in range(len(xys)-1):
                    s0 = xys[i]
                    s1 = xys[i+1]
                    d = dist_point_to_segment(p, s0, s1)
                    if d <= self.max_ds:
                        self.poly.xy = np.array(
                            list(self.poly.xy[:i+1]) +
                            [pd] +
                            list(self.poly.xy[i+1:]))
                        self._update_line()
                        break
        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        ignore = (not self.showverts or event.inaxes is None or
                  event.button != 1 or self._ind is None)
        if ignore:
            return
        x,y = event.xdata, event.ydata

        if self._ind == 0 or self._ind == self.last_vert_ind:
            self.poly.xy[0] = x,y
            self.poly.xy[self.last_vert_ind] = x,y
        else:
            self.poly.xy[self._ind] = x,y
        self._update_line()

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        self.verts = self.poly.xy
        self.last_vert_ind = len(self.poly.xy) - 1
        self.line.set_data(list(zip(*self.poly.xy)))

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >= self.max_ds:
            ind = None
        return ind


def default_vertices(ax):
    """Default to rectangle that has a quarter-width/height border."""
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    w = np.diff(xlims)
    h = np.diff(ylims)
    x1, x2 = xlims + w // 4 * np.array([1, -1])
    y1, y2 = ylims + h // 4 * np.array([1, -1])
    return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))







import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import scipy

img = data.astronaut()
ax = plt.subplot(111)
ax.imshow(img)

mc = MaskCreator(ax)
plt.show()

print(mc.verts)



img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T
print(init.shape)
init = mc.verts

# exit()
snake = active_contour(
    # img,
    gaussian(img, 3),
    init, alpha=0.015, beta=10, gamma=0.001
)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()

exit()
