"""

"""
import glob
import io
import os
import numpy as np
import pandas as pd

import constants

_print = print
# def print(*args):
    # pass

def cache_to_self(met):
    s = '_' + met.__name__ + '_cache'
    def _f(self, *args, **kwargs):
        if not hasattr(self, s):
            setattr(self, s, met(self, *args, **kwargs))
        return getattr(self, s)
    return _f


class M2:

    def __init__(self, path):


        # def print(*args):
        #     pass


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
            # if a0 == 0x84:
                # assert s <= 2, (self.name, s)

            sl = slice(a1, a1 + s * so)
            assert np.all(mask[sl] == 0), (
                f'{a1:#x}',
                set(
                    f'{v:#x}'
                    for v in mask[sl][mask[sl] != 0]
                )
            )
            mask[sl] = a0
            bcount = s * so

            if a0 != 0x84:
                print(f'm2arr at {a0:#5x} pointing to [{a1:#7x}:{a1 + so * s:#7x}] '
                      f'for {s:3} items of {so:2} bytes each. ({bcount:5} bytes total)')
            else:
                print(f'm2arr at {a0:#5x} pointing to [{a1:#7x}:{a1 + so * s:#7x}] '
                      f'for {s:3} items of {so:2} bytes each. ({bcount:5} bytes total) '
                      + ' '.join(
                          f'{i:016b}'
                          for i in self.pull_u16s(a1, bcount // 2)
                      )
                )


        print('First part')
        ff(0x1c, 1)
        ff(0x24, 2)
        ff(0x2c, 4)
        ff(0x34, 1)
        ff(0x3c, 2)
        ff(0x44, 12 * 4) # Vertices
        ff(0x4c, 44) # Skin profiles

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

        print('Second part')
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
        for i in range(s):
        # i = s - 1
        # if True:
            print(f'Skin profile {i}')
            a1 = a0 + i * 44
            ff(a1 + 0x00, 2)

            arr = self.m2array(a1 + 0x00, 2)
            asu8 = np.frombuffer(arr, 'uint16')

            ff(a1 + 0x08, 2)

            arr = self.m2array(a1 + 0x08, 2)
            asu8 = np.frombuffer(arr, 'uint16')

            ff(a1 + 0x10, 4)
            ff(a1 + 0x18, 32) # Submeshes
            ff(a1 + 0x20, 24)
            print(self.pull_u32s(a1 + 0x24))

        mask = (mask != 0)
        print(mask.mean())
        print(mask.sum(), len(self.bts))

    # Offsets *********************************************************************************** **
    def arr_entry_views(self, addr, byte_per_entry):
        entry_count, addr0 = self.pull_u32s(addr, 2)
        return [
            self.mem_view(addr0 + i * byte_per_entry, byte_per_entry)
            for i in range(entry_count)
        ]

    def arr_view(self, addr, byte_per_entry):
        entry_count, addr0 = self.pull_u32s(addr, 2)
        return self.mem_view(addr0, byte_per_entry * entry_count)

    def mem_view(self, addr, byte_count):
        bts = self.bts[addr:addr+byte_count]
        return pd.Series({
            'addr': addr,
            'byte_count': byte_count,
            'bts': bts,
            'asu8': np.frombuffer(bts, 'uint8'),
            'asu16': np.frombuffer(bts[:len(bts) // 2 * 2], 'uint16'),
            'asu32': np.frombuffer(bts[:len(bts) // 4 * 4], 'uint32'),
            'asfloat': np.frombuffer(bts[:len(bts) // 4 * 4], 'float32'),
        })

    @property
    @cache_to_self
    def last_lod(self):
        """Return a list of data about each `G.skin_profiles[-1].submeshes[i]` for each `i`.

        G.skin_profiles[i].submeshes[i].skinSectionId # https://wowdev.wiki/M2/.skin#Mesh_part_ID
        G.skin_profiles[i].batches[i].materialIndex https://wowdev.wiki/M2/.skin#Texture_units
        G.skin_profiles[i].batches[i].skinSectionIndex https://wowdev.wiki/M2/.skin#Submeshes
        G.materials[i].flags & 0x4 https://wowdev.wiki/M2#Render_flags
        """

        lod_v = self.arr_entry_views(0x4c, 0x2c)[-1]
        pts_idxs_v = self.arr_view(lod_v.addr + 0x0, 2)
        faces_idxs_v = self.arr_view(lod_v.addr + 0x8, 2)
        submesh_vs = self.arr_entry_views(lod_v.addr + 0x18, 32)
        texunit_vs = self.arr_entry_views(lod_v.addr + 0x20, 24)
        renderflags_vs = self.arr_entry_views(0x84, 4)

        texunit_v_per_submesh_idx = {
            texunit_v.asu16[2]: texunit_v
            for texunit_v in texunit_vs
        }

        assert len(submesh_vs) <= len(texunit_vs), (len(submesh_vs), len(texunit_vs))

        faces = pts_idxs_v.asu16[faces_idxs_v.asu16]
        res = []
        print('faces', faces.shape,
              'pts', pts_idxs_v.asu16.shape,
              'submesh_vs', len(submesh_vs),
              'texunit_vs', len(texunit_vs),
              'renderflags_vs', len(renderflags_vs),
        )

        for i, submesh_v in enumerate(submesh_vs):

            pts_slice = slice(
                int(submesh_v.asu16[2]),
                int(submesh_v.asu16[2]) + int(submesh_v.asu16[3]),
            )
            faces_slice = slice(
                int(submesh_v.asu16[4]) | (int(submesh_v.asu16[1]) << 16),
                (int(submesh_v.asu16[4]) | (int(submesh_v.asu16[1]) << 16)) + int(submesh_v.asu16[5]),
            )

            # print(submesh_v)
            # if submesh_v.asu16[0] >= len(texunit_vs):
                # _print(f'skinSectionId is {submesh_v.asu16[0]}, how to render {self.name}??')
                # continue
            # texunit_v = texunit_vs[submesh_v.asu16[0]]
            texunit_v = texunit_v_per_submesh_idx[i]
            render_flags = renderflags_vs[texunit_v.asu16[5]].asu16[0]

            print(f'  submesh{i}, render_flags:{render_flags:#b}, pts-slice:{pts_slice}, faces_slice:{faces_slice}')

            assert pts_slice.stop <= pts_idxs_v.asu16.size
            assert faces_slice.stop <= faces.size

            res.append(pd.Series({
                'pts_idxs': faces[faces_slice].reshape(-1, 3),
                'render_flags': render_flags,
            }))
        if submesh_vs:
            assert pts_slice.stop == pts_idxs_v.asu16.size
            assert faces_slice.stop == faces.size

        return res

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
            'tex0': list(asfloat[:, 8:10]),
            'tex1': list(asfloat[:, 10:12]),
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

    def pull_u16s(self, addr, shape=()):
        a = np.frombuffer(self.bts, 'uint16', np.prod(shape, dtype=int), addr)
        a = a.reshape(shape)
        return a

    def pull_floats(self, addr, shape=()):
        a = np.frombuffer(self.bts, 'float32', np.prod(shape, dtype=int), addr)
        a = a.reshape(shape)
        return a

    # ******************************************************************************************* **

if __name__ == '__main__':
    for path in [
            # 'Y:\\model.mpq\\LShoulder_Plate_C_05.m2',
            # 'Y:\\model.mpq\\DarkshoreRuinPillar03.m2', # 52v
            # 'Y:\\model.mpq\\EyeOfKilrog.m2',
            # 'Y:\\model.mpq\\LandMine01.m2', # 48v, 84f
            # 'Y:\\model.mpq\\Chicken.m2',
            # 'Y:\\model.mpq\\Buckler_Round_A_01.m2', # 21v, 40f
            # 'Y:\\model.mpq\\PlaqueBronze02.m2',
            # 'Y:\\model.mpq\\G_FishingBobber.m2',


            p for p in glob.glob('Y:\\model.mpq\\*.m2')
            # if 'KelT' in p
            # if 'Banshee' in p
    ]:
        print('////////////////////////////////////////////////////////////////////////////////')
        print(path)
        m = M2(path)
        # df = m.last_lod.shape
        print(m.last_lod)
        # print(m.vertices)
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
