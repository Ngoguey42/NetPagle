import os
import itertools

import numpy as np
import mss
import matplotlib.pyplot as plt
import matplotlib.cm
import pandas as pd
import shapely.geometry as sg

from show import patchify_points, patchify_polys
from m2 import M2
from wow import WoW
from cam import Camera
from objects import GameObject
from constants import Offset, SCREEN_SIZE, MAGIC_SCALE_FACTOR

w = WoW()
cam = Camera(w)

rows = []
jj = -1

ads = sorted(set([
    f'{i + j * 4:#05x}'
    for i in [0x009c4, 0x009c8, 0x009f8, 0x009fc, 0x00a04, 0x00a08, 0x00a10, 0x00a14, 0x00c70, 0x00c94, 0x00c98]
    for j in range(-5, 6)
]), key=lambda x: int(x, 16))

df = globals().get('df', pd.DataFrame())
print(df.shape)

for p in list(w.gen_players()):
    continue # DEBUG!!

    print(f'{p.addr:#x}')
    # if p != 0x16138008:
        # continue
    # print('OK!!')

    # level: 0x1df8

    # print(w.pull_u32s(addr, (200,)))
    o = 4 * 50 * 0
    # o = 4 * 50 * 12
    n = 2500
    # a =
    # print(p.addr + 0x8, (1,))
    # addr2 = int(w.pull_u32s(p.addr + 0x8, ()))
    # print(addr2, f'{addr2 - p.addr:x}')
    addr2 = p.addr
    # print(' ', w.pull_u32s(addr2, (3,)))
    print(f'  {w.pull_u32s(p.addr + 0x8, ()):#x}',
          f'  {w.pull_u32s(p.addr + 0x8, ()) - p.addr:#x}',
          f'  {int(w.pull_u32s(p.addr + 0x18, ()))}'
          f'  {int(w.pull_u32s(p.addr + 0x1df8, ()))}'
    )

    # if int(w.pull_u32s(p.addr + 0x18, ())) != 0x76005:
    #     continue
    # print('  ok')

    def _pick(i, f):
        # return f'F {f:<+11.3f}'
        if i == 0:
            return f'i {i:<11}'
        if abs(f) > 0.0001 and abs(f) < 10000:
            return f'F {f:<+11.3f}'
        if i > 0 and i < 1000:
            return f'i {i:<11}'
        return f'? {i:<#11x}'

    rows.append({
        f'{i * 4 + o:#07x}': _pick(a, b)
        for i, (a, b) in enumerate(zip(
                w.pull_u32s(addr2 + o, (n,)),
                w.pull_floats(addr2 + o, (n,)),
        ))
    })
    # a
    # rows.append({
    #     f'{i * 4 + o:#05x}': float(v)
    #     for i, v in enumerate(a)
    # })

df = pd.concat([df, pd.DataFrame(rows).T], axis=1)
df = df.T.reset_index(drop=True).T

print(df.shape)

# # print(df)
