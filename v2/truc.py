"""
Made for wow 1.12.1.5875, python>=3.6

````sh
$ pip install pypiwin32==224 Pymem==1.0 psutil mss numpy matplotlib
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
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-2.html#post2331747
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-4.html#post2716691
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-13.html#post3286690
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-22.html#post3450098
https://wowdev.wiki/Enumeration_%26_Structures
https://github.com/acidburn974/CorthezzWoWBot/blob/master/BotTemplate/Constants/Offsets.cs
http://www.cnblogs.com/hmmcsdd/archive/2007/11/30/mangoscharacterdatafielddesc.html

# world to screen
### First attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/599004-world-screen.html#post3661355

### Second attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/535748-world-screen.html#post3347571

### Third Attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/271612-world-screen.html#post1754193

### Misc
https://wowdev.wiki/Common_Types#stringref
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-40.html#post4022185
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/544312-12-1-2016-bans-3.html#post3386119

### MPQ files
https://wow.curseforge.com/projects/mpqviewer/files # Have game closed

##### M2 files
https://wow.curseforge.com/projects/project-3043 # Have game opened
G.skin_profiles[i].batches[i].materialIndex https://wowdev.wiki/M2/.skin#Texture_units
G.skin_profiles[i].batches[i].skinSectionIndex https://wowdev.wiki/M2/.skin#Submeshes
G.materials[i].flags & 0x4 https://wowdev.wiki/M2#Render_flags
G.skin_profiles[i].submeshes[i].skinSectionId # https://wowdev.wiki/M2/.skin#Mesh_part_ID

##### DBC files
https://wowdev.wiki/DB/GameObjectDisplayInfo
https://wowdev.wiki/DB/CreatureModelData



http://www.pudn.com/Download/item/id/101052.html


"""
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

# *********************************************************************************************** **
# *********************************************************************************************** **
# exit()
# it = list(w.gen_game_objects())
it = list(w.gen_players())

print('  Snapping...')
with mss.mss() as sct:
    monitor = sct.monitors[1]
    img = sct.grab(monitor)
    img = np.asarray(img)[:, :, [2, 1, 0]]
    print('  Snaped!', img.shape)

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img)

for go in it:
    bl = [
        # 'Chaise en',
    ]
    wl = [
        # 'lettres'
        # 'Flot'
    ]
    if bl and any(s in go.name for s in bl):
        continue
    if wl and not any(s in go.name for s in wl):
        continue

    xyz = (0, 0, 0, 1)
    xyz = xyz @ go.model_matrix
    assert xyz[-1] == 1

    x, y, z, _ = xyz
    ox = x - cam.xyz[0]
    dx = ['south', 'north'][int(ox > 0)]
    oy = y - cam.xyz[1]
    dy = ['east', 'west'][int(oy > 0)]

    (x, y), visible, behind = cam.world_to_screen(xyz[:3])


    if not behind:
        jj += 1

        mn = str(go.model_name).split("\\")[-1:]
        print(
            f'{jj:2}{go.name:>50}: ('
            f'{ox:+7.2f}({dx})({xyz[0]:<10.2f}), '
            f'{oy:+7.2f}({dy})({xyz[1]:<10.2f}), '
            f'{xyz[2] - cam.xyz[2]:+7.2f}({xyz[2]:<6.2f})) '
            f'{(go.angle / np.pi * 180 + 360) % 360:5.1f}deg {mn}'
        )
        ax.text(x, y, jj, fontsize=10)

        # if not visible:
            # continue
        # if int(w.pull_u32s(go.addr + 0x18, ())) != 0x76005: # Myself
            # continue
        # print('Rendering')

        # for i in range(3):
        #     c = ['red', 'green', 'blue'][i]
        #     for v in np.linspace(0, 1, 15):
        #         xyz = np.r_[(np.arange(3) == i) * v, 1]
        #         xyz = xyz @ go.model_matrix
        #         assert xyz[-1] == 1
        #         (x, y), visible, behind = cam.world_to_screen(xyz[:3])
        #         if not behind:
        #             ax.add_patch(*patchify_points(
        #                 [sg.Point(x, y),],
        #                 radius=2.,
        #                 fill=False,
        #                 ec=c,
        #             ))

        if go.model_name is not None:
            path = str(go.model_name).split("\\")[-1]
            # path = 'Banshee.m2'
            # path = 'KelThuzad.m2'

            path = path.replace('.MDX', '.m2').replace('.mdx', '.m2')
            path = os.path.join('Y:\\model.mpq', path)
            if os.path.isfile(path):
                m = M2(path)

                colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(m.last_lod)))
                for submesh, color in zip(m.last_lod, colors):

                    if submesh.mesh_part_id % 100 not in [0, 1]:
                        continue

                    for a, b, c in submesh.pts_idxs:
                        a, avisible, abehind = cam.world_to_screen(
                            (np.r_[m.vertices.xyz[a], 1] @ go.model_matrix)[:3]
                        )
                        if abehind: # hello
                            continue
                        b, bvisible, bbehind = cam.world_to_screen(
                            (np.r_[m.vertices.xyz[b], 1] @ go.model_matrix)[:3]
                        )
                        if bbehind: # hello
                            continue
                        c, cvisible, cbehind = cam.world_to_screen(
                            (np.r_[m.vertices.xyz[c], 1] @ go.model_matrix)[:3]
                        )
                        if cbehind: # hello
                            continue
                        p = sg.Polygon([a, b, c])

                        if submesh.render_flags & 0x4 or not p.exterior.is_ccw:
                            ax.add_patch(*patchify_polys(
                                p,
                                fill=False,
                                hatch=False,
                                ec=color,
                                lw=1,
                                alpha=0.5,
                            ))

plt.show()
