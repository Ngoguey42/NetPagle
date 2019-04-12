"""

"""
import glob
import io
import os
import numpy as np
import pandas as pd

def print(*args):
    pass


def cache_to_self(met):
    s = '_' + met.__name__ + '_cache'
    def _f(self, *args, **kwargs):
        if not hasattr(self, s):
            setattr(self, s, met(self, *args, **kwargs))
        return getattr(self, s)
    return _f

class M2:

    def __init__(self, path):

        self.bts = open(path, 'rb').read()
        assert self.magic == b'MD20', self.magic
        assert self.version == 256, self.version
        print(self.name, len(self.bts), self.fuss)

        m = 0xec
        n = 40
        df = pd.DataFrame({
            'a': ['{:#x}'.format(v + m) for v in np.arange(n) * 4],
            'i': ['{:#x}'.format(v) for v in self.pull_u32s(m, n)],
            'f': self.pull_floats(m, n),
        }).set_index('a')

        mask = np.zeros(len(self.bts), int)
        def ff(a0, so):
            s, a1 = self.pull_u32s(a0, 2)
            sl = slice(a1, a1 + s * so)
            assert np.all(mask[sl] == 0), (
                f'{a1:#x}',
                set(
                    f'{v:#x}'
                    for v in mask[sl][mask[sl] != 0]
                )
            )
            mask[sl] = a0
            print(f'm2arr at {a0:#5x} pointing to [{a1:#7x}:{a1 + so * s:#7x}] '
                  f'for {s:3} items of {so:2} bytes each. ({s * so:5} bytes total)')

        ff(0x1c, 1)
        ff(0x24, 2)
        ff(0x2c, 4)
        ff(0x34, 1)
        ff(0x3c, 2)
        ff(0x44, 12 * 4)
        ff(0x4c, 44)
        ff(0x54, 1)
        ff(0x5c, 1)
        ff(0x64, 1)
        ff(0x6c, 1)
        ff(0x74, 1)
        ff(0x7c, 2)
        ff(0x84, 4)
        ff(0x8c, 2)
        ff(0x94, 2)
        ff(0x9c, 5) # 5?
        ff(0xa4, 2)
        ff(0xac, 2)

        ff(0xec, 2)
        ff(0xf4, 12)
        ff(0xfc, 1)
        ff(0x104, 1)
        ff(0x10c, 2)
        ff(0x114, 1)
        ff(0x11c, 1)
        ff(0x124, 1)
        ff(0x12c, 1)
        ff(0x134, 1)
        ff(0x13c, 1)
        ff(0x144, 1)

        s, a0 = self.pull_u32s(0x4c, 2)
        # for i in range(s):
        i = s - 1
        if True:
            print(f'Skin profile {i}')
            a1 = a0 + i * 44
            ff(a1 + 0x00, 2)

            arr = self.m2array(a1 + 0x00, 2)
            asu8 = np.frombuffer(arr, 'uint16')

            ff(a1 + 0x08, 2)

            arr = self.m2array(a1 + 0x08, 2)
            asu8 = np.frombuffer(arr, 'uint16')

            ff(a1 + 0x10, 4)
            ff(a1 + 0x18, 32)
            ff(a1 + 0x20, 24)
            print(self.pull_u32s(a1 + 0x24))

        mask = (mask != 0)
        print(mask.mean())
        print(mask.sum(), len(self.bts))

    # Offsets *********************************************************************************** **
    @property
    @cache_to_self
    def last_lod(self):
        lod_count, lod0_addr = self.pull_u32s(0x4c, 2)
        lod_addr = lod0_addr + 0x2c * (lod_count - 1)

        pts = np.frombuffer(self.m2array(lod_addr + 0x0, 2), 'uint16')
        faces = np.frombuffer(self.m2array(lod_addr + 0x8, 2), 'uint16').reshape(-1, 3)
        faces = pts[faces]

        return faces

    @property
    def magic(self):
        return self.bts[:4]

    @property
    def version(self):
        return self.pull_u32s(0x4)

    @property
    def name(self):
        return self.cstring_inarr(0x8)

    @property
    def fuss(self):
        return self.pull_u32s(0x10, 3)

    @property
    @cache_to_self
    def vertices(self):
        a = self.m2array(0x44, 12 * 4)
        asfloat = np.frombuffer(a, 'float32').reshape(-1, 12)
        asu8 = np.frombuffer(a, 'uint8').reshape(-1, 12 * 4)

        df = pd.DataFrame({
            'xyz': list(asfloat[:, 0:3]),
            'bonew': list(asu8[:, 3 * 4:3 * 4 + 4]),
            'bonei': list(asu8[:, 3 * 5:3 * 5 + 4]),
            'nor': list(asfloat[:, 5:8]),
            'tex': list(asfloat[:, 9:]),
        })
        return df

    # ******************************************************************************************* **
    def cstring_inarr(self, addr):
        """m2array<char>"""
        n = self.m2array(addr)
        assert n[-1] == 0
        n = n[:-1].decode('utf8')
        assert n.isprintable()
        return n

    def m2array(self, addr, sizeof1=1):
        s, a = self.pull_u32s(addr), self.pull_u32s(addr + 0x4)
        s = s * sizeof1
        assert a + s < len(self.bts)
        return self.bts[a:a+s]

    def pull_u32s(self, addr, shape=()):
        a = np.frombuffer(self.bts, 'uint32', np.prod(shape, dtype=int), addr)
        a = a.reshape(shape)
        return a

    def pull_floats(self, addr, shape=()):
        a = np.frombuffer(self.bts, 'float32', np.prod(shape, dtype=int), addr)
        a = a.reshape(shape)
        return a

    # ******************************************************************************************* **

if __name__ == '__main__':
    print('////////////////////////////////////////////////////////////////////////////////')
    for path in [
            # 'Y:\\model.mpq\\LShoulder_Plate_C_05.m2',
            # 'Y:\\model.mpq\\DarkshoreRuinPillar03.m2', # 52v
            # 'Y:\\model.mpq\\EyeOfKilrog.m2',
            # 'Y:\\model.mpq\\LandMine01.m2', # 48v, 84f
            # 'Y:\\model.mpq\\Chicken.m2',
            # 'Y:\\model.mpq\\Buckler_Round_A_01.m2', # 21v, 40f
            'Y:\\model.mpq\\PlaqueBronze02.m2',

            # p for p in glob.glob('Y:\\model.mpq\\*.m2')
            # if 'KelT' in p
    ]:
        m = M2(path)
        print('////////////////////////////////////////////////////////////////////////////////')
        # df = m.last_lod.shape
        print(m.last_lod)

# raw = open(path, 'rb').read()
# raw = raw[:256]
# print(len(raw))

# cols = {}
# for name, t, n in [
#         # ('b', lambda x: np.frombuffer(x, np.uint8, 1)[0], 1),
#         ('i', lambda x: np.frombuffer(x, np.int32, 1)[0], 4),
#         # ('f', lambda x: np.frombuffer(x, np.float32, 1)[0], 4),
#         # ('d', lambda x: np.frombuffer(x, np.float64, 1)[0], 8),
# ]:
#     vs = {}
#     for i in range(0, len(raw), n):
#         vs[i] = t(raw[i:i+n])

#     cols[name] = vs
# df = pd.DataFrame(cols)
