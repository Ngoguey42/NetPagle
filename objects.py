import numpy as np

from constants import Offset
import constants as con

class Player:

    def __init__(self, w, addr):
        assert w.pm.read_int(addr + Offset.Object.type) == 4
        self.addr = addr

        self.guid = w.pull_u64s(addr + Offset.Object.guid, ())
        self.name = w.get_player_name(self.guid)

        self.xyz = w.pull_floats(addr + Offset.Player.xyz, (3,))
        self.angle = -w.pull_floats(addr + Offset.Player.angle, ())
        self.model_matrix = (
            rot(self.angle, 2) @
            # TODO: Scale? Taurens female scale is written in m2 file
            # TODO: Scale? Test with the troll whichdoctor that make players bigger
            # scale(w.pull_floats(addr + Offset.GameObject.scale, ())) @
            translate(self.xyz) @
            np.eye(4)
        )

        # TODO: Test with a druid
        # TODO: What about a sheeped player?
        # TODO: Test with an invisible character?
        self.display_id, display_id_bis = w.pull_u32s(addr + Offset.Player.display_ids, (2,))
        self.display_id = display_id_bis
        del display_id_bis
        if self.display_id in w.cmd.df.index:
            self.model_name = w.cmd.df.loc[self.display_id, 'model']
        else:
            self.model_name = None

        self.level = w.pull_u32s(addr + Offset.Player.level, ())
        unitflags = w.pull_u32s(addr + Offset.Player.unitflags)
        self.race = con.race_name_of_race_id[(unitflags >> 0) % 256]
        self.class_ = con.class_name_of_class_id[(unitflags >> 8) % 256]
        self.gender = con.gender_name_of_gender_id[(unitflags >> 16) % 256]

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

        # TODO: What is the purpose of that matrix
        # w.pull_floats(addr + Offset.GameObject.unknown_matrix, (4, 4))

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
