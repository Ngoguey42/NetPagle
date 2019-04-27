"""
Made for wow 1.12.1.5875, python>=3.6

````sh
$ pip install pypiwin32==224 Pymem==1.0 psutil mss numpy matplotlib
```

# Links
### Offsets
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/208754-guide-kind-of-how-i-handle-objects.html
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-3.html#post2436167
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-29.html#post3680708
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-38.html#post3859175
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-2.html#post2331747
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-4.html#post2716691
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-13.html#post3286690
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/328263-wow-1-12-1-5875-info-dump-thread-22.html#post3450098
https://wowdev.wiki/Enumeration_%26_Structures
http://www.cnblogs.com/hmmcsdd/archive/2007/11/30/mangoscharacterdatafielddesc.html

### Offsets on github
https://github.com/WowDevs/Fishbot-1.12.1/blob/master/FishBot%201.12.1/Helpers/Offsets.cs
https://github.com/acidburn974/CorthezzWoWBot/blob/master/BotTemplate/Constants/Offsets.cs
https://github.com/acidburn974/CorthezzWoWBot/blob/master/BotTemplate/Objects/UnitObject.cs
https://github.com/tomcook82/Aesha/blob/master/src/Aesha.Objects/Infrastructure/Offsets.cs
https://github.com/wancharle/winprocess-utils/blob/master/test.js
https://github.com/tovobi/cppWow1/blob/master/cppWow1/Pointers.cpp
https://github.com/tovobi/Revlex6/blob/master/Revlex6/WowPointers.h
https://github.com/tovobi/Revlex/blob/master/Revlex/Pointers.cs

https://github.com/zhaoleirs/ccbot/blob/master/ThadHack/Constants/Offsets.cs
https://github.com/Nekkidso/ZzukBot_v1/blob/master/ThadHack/Constants/Offsets.cs
https://github.com/Icesythe7/ZzukBot_V3_NoAuth/blob/master/ZzukBot_WPF/Constants/Offsets.cs
https://github.com/Zz9uk3/ZzukBot_v1/blob/master/ThadHack/Constants/Offsets.cs
https://github.com/Zz9uk3/ZzukBot_V3/blob/master/ZzukBot_WPF/Constants/Offsets.cs
https://github.com/Sterioss/WoWMemory/blob/master/offset.py
https://github.com/Zz9uk3/ClassicFramework/blob/master/ClassicFramework/Objects/WoWPlayer.cs

### world to screen
##### First attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/599004-world-screen.html#post3661355

##### Second attempt
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/535748-world-screen.html#post3347571

##### Third Attempt
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
https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-emulator-servers/wow-emu-guides-tutorials/159134-race-class-gender-ids-ascent.html
https://shynd.wordpress.com/


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

# TODO: RENDER in argv
# RENDER = False
RENDER = True

w = WoW()
cam = Camera(w)

it = []
it += list(w.gen_game_objects())
it += list(w.gen_players())

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
jj = 0

for go in it:
    bl = [
        # 'Chaise en',
    ]
    wl = [
        # 'lettres'
        # 'Flot'
    ]
    if bl and any(s.lower() in go.name.lower() for s in bl):
        continue
    if wl and not any(s.lower() in go.name.lower() for s in wl):
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

    # render = visible
    render = not behind
    # render = int(w.pull_u32s(go.addr + 0x18, ())) == 0x76005: # Myself
    mn = str(go.model_name).split("\\")[-1:]

    if (not RENDER) or render:
        jj += 1
        print(
            f'{jj:2}{go.name:>50}: ('
            f'{ox:+7.2f}({dx})({xyz[0]:<10.2f}), '
            f'{oy:+7.2f}({dy})({xyz[1]:<10.2f}), '
            f'{xyz[2] - cam.xyz[2]:+7.2f}({xyz[2]:<6.2f})) '
            f'{(go.angle / np.pi * 180 + 360) % 360:5.1f}deg {mn}'
        )


    if not (RENDER and render):
        continue

    ax.text(x, y, jj, fontsize=10)

    # # x, y, z axes
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

    if go.model_name is None:
        continue

    path = str(go.model_name).split("\\")[-1]
    # path = 'Banshee.m2'
    # path = 'KelThuzad.m2'

    path = path.replace('.MDX', '.m2').replace('.mdx', '.m2')
    path = os.path.join('Y:\\model.mpq', path)

    if not os.path.isfile(path):
        print("  Can't render, missing file".format())
        continue

    m = M2(path)

    submeshes = [
        submesh
        for submesh in m.last_lod
        if submesh.mesh_part_id % 100 in [0, 1]
    ]
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(submeshes)))

    for submesh, color in zip(submeshes, colors):
        for a, b, c in submesh.pts_idxs:
            a, avisible, abehind = cam.world_to_screen(
                (np.r_[m.vertices.xyz[a], 1] @ go.model_matrix)[:3]
            )
            if abehind:
                continue
            b, bvisible, bbehind = cam.world_to_screen(
                (np.r_[m.vertices.xyz[b], 1] @ go.model_matrix)[:3]
            )
            if bbehind:
                continue
            c, cvisible, cbehind = cam.world_to_screen(
                (np.r_[m.vertices.xyz[c], 1] @ go.model_matrix)[:3]
            )
            if cbehind:
                continue
            p = sg.Polygon([a, b, c])

            if submesh.render_flags & 0x4 or not p.exterior.is_ccw:
                ax.add_patch(*patchify_polys(
                    p,
                    fill=False,
                    hatch=False,
                    ec=color,
                    lw=1,
                    alpha=0.3,
                ))

if RENDER:
    plt.show()
