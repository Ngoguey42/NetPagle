
import shapely.geometry as sg
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PATHPATCH_PROPS = list(mpatches.PathPatch(None).properties().keys())
CIRCLEPOLY_PROPS = list(mpatches.CirclePolygon((0,0)).properties().keys())
TEXT_PROPS = list(plt.Text().properties().keys())

def patchify_points(geoms, ec='black', fc='black', radius=0.1, *args, **kwargs):
    """Creates points patches from a sequence of geometries.
    Radius should be set accordingly to final picture extent.
    Default style can be modified using same kwargs as matplotlib.patches.CirclePolygon"""

    patches = []
    for point in _point_iterator(geoms):
        if point.is_empty:
            continue
        # Monkey patching to easily add some basic styling
        kwargs_ = kwargs_from_prop(point, CIRCLEPOLY_PROPS, ['ec', 'fc', 'radius'])
        kwargs_.update(kwargs)
        ec = getattr(point, 'ec', ec)
        fc = getattr(point, 'fc', fc)
        radius = getattr(point, 'radius', radius)
        xy = point.coords
        patch = mpatches.CirclePolygon(
            *args,
            xy=tuple(xy)[0],
            radius=radius,
            ec=ec,
            fc=fc,
            **kwargs_
            )
        patches.append(patch)
    return patches

def _poly_iterator(obj):
    if isinstance(obj, (sg.Polygon)):
        yield obj
    elif isinstance(obj, (sg.MultiPolygon)):
        for obj2 in obj.geoms:
            yield obj2
    elif isinstance(obj, (sg.LineString)):
        yield sg.Polygon(obj)
    elif isinstance(obj, sg.MultiPoint):
        yield sg.Polygon(obj)
    else:
        try:
            tup = tuple(obj)
        except TypeError:
            raise TypeError('Could not use type %s' % type(obj))
        else:
            for obj2 in tup:
                for poly in _poly_iterator(obj2):
                    yield poly

def patchify_polys(polys, hatch='\\', ec='red', lw=2, fill=False, *args, **kwargs):
    """Creates patches from sequence of polygons.
    Does not take into account holes in polygon.
    Default style can be modified using ec (edge color), lw (linewidth), fill,
    or other kwargs from matplotlib.patches.Polygon"""

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by Shapely's
        # analytic methods have the proper coordinate order, no need to sort.

        def ring_coding(ob):
            n = len(ob.coords)
            codes = np.ones(n, dtype=mpath.Path.code_type) * mpath.Path.LINETO
            codes[0] = mpath.Path.MOVETO
            return codes

        vertices = [polygon.exterior] + list(polygon.interiors)
        for i in range(len(vertices)):
            should_be_ccw = i == 0
            is_ccw = vertices[i].is_ccw

            vertices[i] = np.asarray(vertices[i])
            if is_ccw != should_be_ccw:
                vertices[i] = np.flipud(vertices[i])

        vertices = np.concatenate(vertices)

        codes = np.concatenate(
            [ring_coding(polygon.exterior)]
            + [ring_coding(r) for r in polygon.interiors])

        return mpath.Path(vertices, codes)

    patches = []
    for poly in _poly_iterator(polys):
        if poly.is_empty:
            continue
        path = pathify(poly)
        # Monkey patching to easily add some basic styling
        kwargs_ = kwargs_from_prop(poly, PATHPATCH_PROPS, ['ec', 'lw', 'hatch', 'fill'])
        kwargs_.update(kwargs)
        ec = getattr(poly, 'ec', ec)
        lw = getattr(poly, 'lw', lw)
        hatch = getattr(poly, 'hatch', hatch)
        fill = getattr(poly, 'fill', fill)
        patch = mpatches.PathPatch(
            *args,
            path=path,
            fill=fill,
            hatch=hatch,
            ec=ec,
            lw=lw,
            **kwargs_,
        )
        patches.append(patch)
    return patches

def _point_iterator(obj):
    if isinstance(obj, (sg.Point)):
        yield obj
    elif isinstance(obj, (sg.MultiPoint)):
        for p in obj.geoms:
            yield p
    elif isinstance(obj, (sg.LineString)):
        for c in obj.coords:
            yield sg.Point(c)
    elif isinstance(obj, (sg.MultiLineString)):
        for l in obj.geoms:
            yield from _point_iterator(l)
    elif isinstance(obj, (sg.Polygon)):
        yield from _point_iterator(sg.LineString(obj.exterior))
        for obj2 in obj.interiors:
            yield from _point_iterator(sg.LineString(obj2))
    else:
        try:
            tup = tuple(obj)
        except TypeError:
            raise TypeError('Could not use type %s' % type(obj))
        else:
            for obj2 in tup:
                yield from _point_iterator(obj2)
def kwargs_from_prop(geom, props, special_cases):
    return {
        key: getattr(geom, key, 'not here')
        for key in props
        if getattr(geom, key, 'not here') != 'not here' and key not in special_cases
    }
