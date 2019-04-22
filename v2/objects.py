import numpy as np

from constants import Offset

class Player:
    def __init__(self, w, addr):
        assert w.pm.read_int(addr + Offset.Object.type) == 4
        self.addr = addr

        self.xyz = w.pull_floats(addr + 0x009b8, (3,))
        self.angle = -w.pull_floats(addr + 0x009c4, ())

        self.model_matrix = (
            rot(self.angle, 2) @
            # TODO: Scale? At least for taurens female
            # scale(w.pull_floats(addr + Offset.GameObject.scale, ())) @
            translate(self.xyz) @
            np.eye(4)
        )

        self.display_id, display_id_bis = w.pull_u32s(addr + 0x1f7c, (2,))
        assert self.display_id == display_id_bis, (self.display_id, display_id_bis)
        del display_id_bis

        level = w.pull_u32s(addr + 0x01df8, ())
        race = (w.pull_u32s(addr + 0x01e00, ()) >> 0) % 255
        gender = (w.pull_u32s(addr + 0x01e00, ()) >> 16) % 255

        if self.display_id in w.cmd.df.index:
            self.model_name = w.cmd.df.loc[self.display_id, 'model']
        else:
            self.model_name = None

        self.name = 'idk' # TODO: Find name!

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
        self.angle = w.pull_floats(addr + Offset.GameObject.angle, ())
        self.quaternion = w.pull_floats(addr + Offset.GameObject.quaternion, (4,)) # i, j, k, real

        self.display_id = w.pm.read_uint(addr + Offset.GameObject.display_id)
        if self.display_id in w.godi.df.index:
            self.model_name = w.godi.df.loc[self.display_id, 'model']
        else:
            self.model_name = None

        q = self.quaternion * [-1, -1, -1, 1]
        rot_matrix = np.asarray([

            [1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
             2 * q[0] * q[1] - 2 * q[2] * q[3],
             2 * q[0] * q[2] + 2 * q[1] * q[3],
             0],

            [2 * q[0] * q[1] + 2 * q[2] * q[3],
             1 - 2 * q[0] ** 2 - 2 * q[2] ** 2,
             2 * q[1] * q[2] - 2 * q[0] * q[3],
             0],

            [2 * q[0] * q[2] - 2 * q[1] * q[3],
             2 * q[1] * q[2] + 2 * q[0] * q[3],
             1 - 2 * q[0] ** 2 - 2 * q[1] ** 2,
             0],
            [0, 0, 0, 1],
        ])
        trans_mat = w.pull_floats(addr + Offset.GameObject.unknown_matrix, (4, 4))

        self.model_matrix = (
            rot_matrix @
            scale(w.pull_floats(addr + Offset.GameObject.scale, ())) @
            translate(self.xyz) @
            np.eye(4)
        )

def rot(angle, axis):
    m = np.eye(4)
    for i in {0, 1, 2} - {axis}:
        m[i, i] = np.cos(angle)
    if axis == 0:
        m[1, 2] = -np.sin(angle)
        m[2, 1] = np.sin(angle)
    elif axis == 1:
        m[0, 2] = np.sin(angle)
        m[2, 0] = -np.sin(angle)
    elif axis == 2:
        m[0, 1] = -np.sin(angle)
        m[1, 0] = np.sin(angle)
    else:
        assert False
    return m

def translate(xyz):
    m = np.eye(4)
    m[3, :3] = xyz
    return m

def scale(f):
    m = np.eye(4) * f
    m[-1, -1] = 1
    return m
