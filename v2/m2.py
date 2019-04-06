"""

"""
import glob
import io
import os
import numpy as np
import pandas as pd


class M2:

    def __init__(self, path):
        self.bts = open(path, 'rb').read()
        assert self.magic == b'MD20', self.magic
        assert self.version == 256, self.version
        print(self.name, len(self.bts), self.fuss)
        # assert np.all(self.fuss == [0, 0, 0]), self.fuss
        # print(v)

        m = 0xec
        n = 40
        df = pd.DataFrame({
            'a': ['{:#x}'.format(v + m) for v in np.arange(n) * 4],
            'i': ['{:#x}'.format(v) for v in self.pull_u32s(m, n)],
            'f': self.pull_floats(m, n),
        }).set_index('a')
        # print(df)


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
        ff(0x44, 12)
        # print(self.pull_floats(self.pull_u32s(0x48), (52, 12)))
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


    # Offsets *********************************************************************************** **
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
            'Y:\\model.mpq\\Chicken.m2', # 21v, 40f
            # 'Y:\\model.mpq\\Buckler_Round_A_01.m2', # 21v, 40f

            # p for p in glob.glob('Y:\\model.mpq\\*.m2')
            # if 'KelT' in p
    ]:
        m = M2(path)
        print('////////////////////////////////////////////////////////////////////////////////')
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
