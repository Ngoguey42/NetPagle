"""
Made for wow 1.12.1.5875, python>=3.6


````sh
$ pip install pypiwin32==224 Pymem==1.0 psutil mss numpy
```

# Links

### Fishbot
https://github.com/WowDevs/Fishbot-1.12.1/blob/fd3855845ae12e32ca5477526017b0b9ee680b9c/FishBot%201.12.1/GUI/MainWindow.cs
https://github.com/WowDevs/Fishbot-1.12.1/blob/fd3855845ae12e32ca5477526017b0b9ee680b9c/FishBot%201.12.1/Helpers/Offsets.cs?ts=4
https://github.com/WowDevs/Fishbot-1.12.1/blob/fd3855845ae12e32ca5477526017b0b9ee680b9c/FishBot%201.12.1/Hook/Hook.cs?ts=4

### WowObjects / offsets
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/208754-guide-kind-of-how-i-handle-objects.html
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-3.html#post2436167
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-29.html#post3680708
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-38.html#post3859175

# world to screen
### First attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/599004-world-screen.html#post3661355

### Second attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/535748-world-screen.html#post3347571

### Third Attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/271612-world-screen.html#post1754193

### Misc
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-40.html#post4022185
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/544312-12-1-2016-bans-3.html#post3386119

"""

import functools

import numpy as np
import pymem
import psutil
import mss
import skimage.io

MAGIC_SCALE_FACTOR = 1 / 1.10

def set_pretty_print_env(level=None):
    import logging
    import numpy as np
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
    player_name = 0x827D88
    # target_guid = 0x74E2D4
    obj_manager = 0x00741414
    version = 0x00837C0
    camera = 0x0074B2BC

    class ObjectManager:
        first_obj = 0xAC

    class Object:
        type = 0x14
        next = 0x3C

    class GameObject:
        guid = 0x30
        name1 = 0x214
        name2 = 0x8
        xyz = 0x248

    class Camera:
        offset = 0x65B8
        xyz = 0x8
        facing = xyz + 3 * 4
        fov = xyz + 14 * 4
        aspect = xyz + 15 * 4

class GameObject:
    def __init__(self, w, addr):
        assert w.pm.read_int(addr + Offset.Object.type) == 5
        self.addr = addr
        a = addr
        a += Offset.GameObject.name1
        a = w.pm.read_uint(a)
        a += Offset.GameObject.name2
        a = w.pm.read_uint(a)
        self.name = w.pm.read_string(a)
        self.xyz = w.pull_floats(addr + Offset.GameObject.xyz, (3,))

class Camera():
    def __init__(self, w):
        cam0 = w.pm.read_uint(w.base_address + Offset.camera)
        cam1 = w.pm.read_uint(cam0 + Offset.Camera.offset)

        self.xyz = w.pull_floats(cam1 + Offset.Camera.xyz, (3,))
        print(self.xyz)
        self.facing = w.pull_floats(cam1 + Offset.Camera.facing, (3, 3))
        print(self.facing)
        self.fov = w.pull_floats(cam1 + Offset.Camera.fov, ())
        self.aspect = w.pull_floats(cam1 + Offset.Camera.aspect, ())
        self.size = np.asarray([1920, 1080]) # TODO: Find offsets
        assert np.allclose(self.aspect, np.divide.reduce(self.size.astype(float)))

    def world_to_screen(self, xyz):
        diff = xyz - self.xyz
        view = diff @ np.linalg.inv(self.facing)
        cam = np.asarray([-view[1], -view[2], view[0]])

        behind = cam[-1] < 0

        fx = (1 / (1 + 1 / self.aspect ** 2)) ** 0.5
        fy = fx / self.aspect

        fx = np.tan(self.fov * fx / 2 * MAGIC_SCALE_FACTOR)
        fy = np.tan(self.fov * fy / 2)

        fx = cam[0] / (fx * cam[2])
        fy = cam[1] / (fy * cam[2])

        outside = np.any(np.abs([fx, fy]) > 1) or behind

        fx = self.size[0] / 2 * (1 + fx)
        fy = self.size[1] / 2 * (1 + fy)

        return np.asarray([fx, fy]), outside, behind

class WoW:
    def __init__(self, pid=None):
        if pid is None:
            pid, = [ps.pid for ps in psutil.process_iter() if ps.name() == 'WoW.exe']

        pm = pymem.Pymem()
        pm.open_process_from_id(pid)
        self.pid = pid
        self.pm = pm

        mod, = [mod for mod in pm.list_modules() if mod.name == 'WoW.exe']
        base_address = mod.lpBaseOfDll
        obj_mgr_addr = pm.read_uint(base_address + Offset.obj_manager)
        first_obj_addr = pm.read_uint(obj_mgr_addr + Offset.ObjectManager.first_obj)

        self.mod = mod
        self.base_address = base_address
        self.player_name = pm.read_string(self.base_address + Offset.player_name)
        self.first_obj_addr = first_obj_addr

    def pull_floats(self, addr, shape):
        a = np.empty(shape, 'float32')
        for i, idxs in enumerate(np.ndindex(*shape)):
            a[idxs] = self.pm.read_float(addr + i * 4)
        return a.astype('float64')

    def gen_objects(self):
        obj_addr = self.first_obj_addr
        while True:
            if not (obj_addr != 0 and obj_addr & 0x1 == 0):
                break
            yield obj_addr
            next_addr = self.pm.read_uint(obj_addr + Offset.Object.next)
            if next_addr == obj_addr:
                break
            obj_addr = next_addr

    def gen_game_objects(self):
        for obj_addr in self.gen_objects():
            if self.pm.read_int(obj_addr + Offset.Object.type) == 5:
                yield GameObject(self, obj_addr)

w = WoW()
cam = Camera(w)
for go in list(w.gen_game_objects()):
    if 'Aller' not in go.name:
        continue

    x, y, z = go.xyz
    ox = x - cam.xyz[0]
    dx = ['south', 'north'][int(ox > 0)]
    oy = y - cam.xyz[1]
    dy = ['east', 'west'][int(oy > 0)]

    print(f'{go.name:>30}: ('
          f'{ox:+7.2f}({dx})({x:<10.2f}), '
          f'{oy:+7.2f}({dy})({y:<10.2f}), '
          f'{z - cam.xyz[2]:+7.2f}({z:<10.2f})), '
    )

    (x, y), outside, behind = cam.world_to_screen(go.xyz)
    print('>', x, y, outside, behind)

    if not outside:
        print('  Snapping...')
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            img = np.asarray(img)[:, :, [2, 1, 0]]
            print('  Snaped!', img.shape)
        for ii in range(-2, 3):
            for jj in range(-2, 3):
                img[int(y - ii), int(x - jj)] = [255, 0, 0]
        print('  Writing...')
        skimage.io.imsave('check.png', img)
        print('  Wrote check.png')

    x = x / cam.size[0]
    y = y / cam.size[1]
    print(f'  {x:.1%}, {y:.1%}')
