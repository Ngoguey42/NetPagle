import os
import itertools

import numpy as np
import mss
import matplotlib.pyplot as plt
import matplotlib.cm
import pandas as pd
import shapely.geometry as sg
import pymem

from show import patchify_points, patchify_polys
from m2 import M2
from wow import WoW
from cam import Camera
from objects import GameObject
import constants

GODI_PATH='Y:\\dbc\\GameObjectDisplayInfo.dbc'
CMD_PATH='Y:\\dbc\\CreatureModelData.dbc'
MODELS_PREFIX='Y:\\model.mpq'
w = WoW(godi_path=GODI_PATH, cmd_path=CMD_PATH)
cam = Camera(w)

rows = []
df = globals().get('df', pd.DataFrame())

# Mem Explorer
DEPTH_MAX = 5
BCOUNT_START = 2500
BCOUNT_IN_DEPTH = 2000
seen = set()
def find_at(addr, bcount, depth=()):
    if addr in seen:
        return
    seen.add(addr)
    if len(seen) % 1000 == 1:
        print(addr, bcount, depth, len(seen))

    bcount = bcount // 4 * 4
    v = w.view(addr, bcount)
    s = v.asascii
    i = s.find(SUBSTR)
    if i != -1:
        print('////////////////////////////////////////////////////////////////////////////////')
        print(depth)
        print(i, f'{i:#x}')
        print('////////////////////////////////////////////////////////////////////////////////')
    for idx, (i, f) in enumerate(zip(v.asu32, v.asfloat)):
        if i == 0:
            continue
        if np.isfinite(f) and abs(f) > 0.0001 and abs(f) < 10000:
            continue
        if i > 0 and i < 1000:
            continue
        i = int(i)
        try:
            w.pull_u32s(i, ())
        except pymem.exception.MemoryReadError:
            continue
        except pymem.exception.WinAPIError:
            print('ola')
            continue
        if len(depth) + 1 < DEPTH_MAX:
            find_at(i, BCOUNT_IN_DEPTH, depth + (
                idx, i,
            ))

for i, p in enumerate(list(w.gen_players())):
    addr2 = p.addr
    print(f'{i:2d}  `{p.name:12}({p.class_:7} {p.race:9} {p.gender:6} '
          f'{p.level:2} {p.guid:#08x})` {p.addr:#x}',
          # f'  {w.pull_u32s(p.addr + 0x8, ()):#x}',
          # f'  {w.pull_u32s(p.addr + 0x8, ()) - p.addr:#x}',
          f'  {int(w.pull_u32s(p.addr + 0x18, ())):#x}'
    )

    def _stringify(i, f):
        if i == 0:
            return f'i {i:<11}'
        if abs(f) > 0.0001 and abs(f) < 10000:
            return f'F {f:<+11.3f}'
        if i > 0 and i < 1000:
            return f'i {i:<11}'
        # TODO: infer pointer, else FLAGS
        return f'? {i:<#11x}'
    o = 0
    n = 3000
    rows.append(pd.Series({
        f'{i * 4 + o:#07x}': _stringify(a, b)
        for i, (a, b) in enumerate(zip(
                w.pull_u32s(addr2 + o, (n,)),
                w.pull_floats(addr2 + o, (n,)),
        ))
    }))

df = pd.concat([df, pd.DataFrame(rows).T], axis=1)
df = df.T.reset_index(drop=True).T
# print(df)
