# 1.12.1.5875
import functools

import numpy as np
import pymem
import psutil
import mss
import skimage.io

def set_pretty_print_env(level=None):
    import logging
    import numpy as np
    import shapely.geometry as sg
    import pandas as pd
    import warnings

    np.set_printoptions(linewidth=250, threshold=np.nan, suppress=True)

    if level is None:
        level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)

    pd.set_option('display.width', 260)
    pd.set_option('display.max_colwidth', 260)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', 25)

    # http://stackoverflow.com/a/7995762
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.INFO, "\033[1;34m%s \033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.DEBUG, "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))

    logging.getLogger('matplotlib').setLevel(logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

set_pretty_print_env()
del set_pretty_print_env

class Offset:
    player_name = 0x827D88 # ok
    target_guid = 0x74E2D4 # bof
    obj_manager = 0x00741414 # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/208754-guide-kind-of-how-i-handle-objects.html

    version = 0x00837C0 # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-29.html#post3680708

    obj_type = 0x14
    obj_guid = 0x30

    object_name1 = 0x214
    object_name2 = 0x8

    # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-3.html#post2436167
    # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-38.html#post3859175
    # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-40.html#post4022185
    # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/599004-world-screen.html#post3661355
    # https://github.com/Ngoguey42/scop/blob/master/srcs/ftmath/matrix4_miscop.c?ts=4#L54
    # https://github.com/Ngoguey42/scop/blob/0aef97c4f1c6f8b6cd263d0c870779a7ec0d83ca/srcs/obs/obs_update.c?ts=4#L36
    # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/271612-world-screen.html
    # https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/535748-world-screen.html
    camera = 0x0074B2BC
    camera_offset = 0x000065B8
    camera_position = 0x8

    x = 0x248
    y = 0x24c
    z = 0x250
    rot = 0x254

class Size:
    obj_manager = 0xAC
    obj = 0x3C

pid, = [ps.pid for ps in psutil.process_iter() if ps.name() == 'WoW.exe']
print(f'WoW.exe pid={pid}')
pm = pymem.Pymem()
pm.open_process_from_id(pid)
print(pm)
mod, = [mod for mod in pm.list_modules() if mod.name == 'WoW.exe']
print(mod)
ba = mod.lpBaseOfDll
print(f'base address {ba:#x}')

def _debug(addr, *args, n, o, what=None, quiet=False):
    if not quiet:
        s = f'> {n} at {addr:#x}'
        if what is not None:
            s += f' ({what})'
        print(s)
    v = o(addr, *args)
    if not quiet:
        if isinstance(v, int):
            s = f'  `{v:#x}`'
        else:
            s = f'  `{v}`'
        print(s)
    return v

for name in dir(pm):
    if not name.startswith('read_'):
        continue
    o = getattr(pm, name)
    setattr(pm, name, functools.partial(_debug, n=name, o=o))
    setattr(pm, f'{name}_q', functools.partial(_debug, n=name, o=o, quiet=True))
    setattr(pm, f'__{name}', o)


pm.read_string(ba + Offset.player_name, what='player name')
# pm.read_string(ba + Offset.version, what='version')
pm.read_ulong(ba + Offset.target_guid, what='target guid')
obj_mgr_addr = pm.read_uint(ba + Offset.obj_manager, what='object manager address')
first_obj_addr = pm.read_uint(obj_mgr_addr + Size.obj_manager, what='first object address')

# cam https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-38.html#post3859175

cam0 = pm.read_uint(ba + Offset.camera, what='camera0')
cam1 = pm.read_uint(cam0 + Offset.camera_offset, what='camera1')
# for i in range(3 + 3 * 3 + 4):
#     v = pm.read_float_q(cam1 + Offset.camera_position + i * 4, what='camera1')
#     print(v)

cx = pm.read_float_q(cam1 + Offset.camera_position + 0, what='camera1')
cy = pm.read_float_q(cam1 + Offset.camera_position + 4, what='camera1')
cz = pm.read_float_q(cam1 + Offset.camera_position + 8, what='camera1')

def pull(addr, shape):
    a = np.empty(shape, 'float32')
    for i, idxs in enumerate(np.ndindex(*shape)):
        a[idxs] = pm.read_float_q(addr + i * 4)
    return a

def create_projection_matrix(fov, ratio, near, far):
    # https://docs.microsoft.com/en-us/windows/desktop/direct3d9/d3dxmatrixperspectivefovrh
    h = 1 / np.tan(fov / 2)
    w = h / ratio
    return np.asarray([
        [w, 0, 0, 0],
        [0, h, 0, 0],
        [0, 0, far / (near - far), -1],
        [0, 0, near * far / (near - far), 0],
    ], 'float32')
    # tmp1 = 1 / np.tan(0.5 * fov)
    # tmp2 = -far / (far - near)
    # m = np.eye(4, dtype='float32')
    # m[0, 0] = tmp1 / ratio
    # m[1, 1] = tmp1
    # m[2, 2] = tmp2
    # m[2, 3] = tmp2 * near * 2
    # m[3, 2] = -1
    # m[3, 3] = 0
    # return m

def create_view_matrix(eye, at, up):
    # https://docs.microsoft.com/en-us/windows/desktop/direct3d9/d3dxmatrixlookatrh
    z = eye - at
    z = z / (z ** 2).sum() ** 0.5

    x = np.cross(up, z)
    x = x / (x ** 2).sum() ** 0.5

    y = np.cross(z, x)
    return np.asarray([
    	[x[0], x[1], x[2], (x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2])],
	    [y[0], y[1], y[2], (y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2])],
	    [z[0], z[1], z[2], (z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2])],
	    [0, 0, 0, 1],
    ],'float32').T
    # return np.asarray([
    # 	[x[0], x[1], x[2], -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2])],
	#     [y[0], y[1], y[2], -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2])],
	#     [z[0], z[1], z[2], -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2])],
	#     [0, 0, 0, 1],
    # ],'float32')


cxyz = pull(cam1 + Offset.camera_position + 0 * 4, (3,))
cfacing = pull(cam1 + Offset.camera_position + 3 * 4, (3, 3))
near = pull(cam1 + Offset.camera_position + 12 * 4, ())
far = pull(cam1 + Offset.camera_position + 13 * 4, ())
fov = pull(cam1 + Offset.camera_position + 14 * 4, ())
aspect = pull(cam1 + Offset.camera_position + 15 * 4, ())
proj = create_projection_matrix(fov, aspect, near, far)

# view = cfacing
# view = create_view_matrix(
#     cxyz,
#     cxyz + [cfacing[0, 0], cfacing[0, 1], cfacing[0, 2]],
#     np.asarray([0, 0, 1.]),
# )

# print('//////////////////////////////////////////////////')
# print('near, far, fov, aspect')
# print(near, far, fov, aspect)
# print('//////////////////////////////////////////////////')
# print('facing')
# print(cfacing)
# print('//////////////////////////////////////////////////')
# print('proj')
# print(proj)
# print('//////////////////////////////////////////////////')
# print('view')
# print(view)

obj_addr = first_obj_addr
while True:
    if not (obj_addr != 0 and obj_addr & 0x1 == 0):
        print(f'stop 0 {obj_addr:#x}')
        break
    type_ = pm.read_int_q(obj_addr + Offset.obj_type, what='object type')
    if type_ == 5:
        # print('////////////////////////////////////////////////////////////////////////////////')
        guid = pm.read_int_q(obj_addr + Offset.obj_guid, what='object guid')

        # s = pm.read_bytes_q(obj_addr, 60000)
        # s = ''.join(map(chr, s))
        # s = ''.join([
        #     chr(c) if chr(c).isprintable() else '-'
        #     for c in s
        # ])
        # print(s)

        a = obj_addr
        a += Offset.object_name1
        a = pm.read_uint_q(a)
        a += Offset.object_name2
        a = pm.read_uint_q(a)
        name = pm.read_string_q(a)

        if all(
                not s in name
                for s in [
                        # 'Flott'
                        # 'Baril d'
                        'Terrest'
                ]
        ):

        # if any(
        #         s in name
        #         for s in [
        #                 'Chaise', 'Le Caprice', 'La Dame M', 'Zepp', 'Feu de cam', 'Feu doui', 'La Bra',
        #                 'Banc en', 'de Mil'
        #         ]
        # ):
            next_addr = pm.read_uint_q(obj_addr + Size.obj, what='next object addr')
            if next_addr == obj_addr:
                print('stop 1')
                break
            obj_addr = next_addr
            continue

        # for i in range(30000 // 4):
        #     x = pm.read_float_q(obj_addr + i * 4, what='x')
        #     y = pm.read_float_q(obj_addr + (i + 1) * 4, what='x')
        #     xy = np.asarray([x, y])

        #     if np.abs(xy[0] - cxy[0]) < 1000:
        #         print(hex(i * 4), xy)
        #         break

        x = pm.read_float_q(obj_addr + Offset.x, what='x')
        y = pm.read_float_q(obj_addr + Offset.y, what='y')
        z = pm.read_float_q(obj_addr + Offset.z, what='z')
        rot = pm.read_float_q(obj_addr + Offset.rot, what='rot')


        ox = x - cxyz[0]
        dx = ['south', 'north'][int(ox > 0)]
        oy = y - cxyz[1]
        dy = ['east', 'west'][int(oy > 0)]

        # print(name, x, y, z, rot)
        print(f'{name:>30}: ('
              f'{ox:+7.2f}({dx})({x:<10.2f}), '
              f'{oy:+7.2f}({dy})({y:<10.2f}), '
              f'{z - cxyz[2]:+7.2f}({z:<10.2f})), '
              f'{rot:12.8g}')
        diff = np.asarray([x, y, z]) - cxyz

        view = diff @ np.linalg.inv(cfacing)
        # view = diff @ np.linalg.inv(cfacing)
        cam = np.asarray([-view[1], -view[2], view[0]])

        fx = (1 / (1 + 1 / aspect ** 2)) ** 0.5
        fy = fx / aspect

        fx = np.tan(fov * fx / 2)
        fy = np.tan(fov * fy / 2)
        fx = 1920 / 2 / fx
        fy = 1080 / 2 / fy

        fx = 1920 / 2 + cam[0] * fx / cam[2]
        fy = 1080 / 2 + cam[1] * fy / cam[2]



        print('Snapping...')
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            print('Snaped!')
            img = np.asarray(img)[:, :, [2, 1, 0]]
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                img[int(fy - ii), int(fx - jj)] = [255, 0, 0]
        print('Writing...')
        skimage.io.imsave('check.png', img)
        print('Wrote check.png')



        fx = fx / 1920
        fy = fy / 1080
        print(f'  {fx:.1%}, {fy:.1%}')



    next_addr = pm.read_uint_q(obj_addr + Size.obj, what='next object addr')
    if next_addr == obj_addr:
        print('stop 1')
        break
    obj_addr = next_addr
#
