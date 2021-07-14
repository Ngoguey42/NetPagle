print("> Importing...")
from enum import Enum
import time
import datetime

import pyautogui

from wow import WoW
from cam import Camera
from constants import Offset

LURE = False
# LURE = True
CPS = 30
FISH = lambda: pyautogui.press("f6")
STOP = lambda: pyautogui.press('space')
LURE_SELECT = lambda: (pyautogui.moveTo(1850, 450, duration=0.5), pyautogui.rightClick())
LURE_APPLY = lambda: (pyautogui.moveTo(1650, 800, duration=0.5), pyautogui.click())
LURE_CONFIRM = lambda: (pyautogui.moveTo(900, 200, duration=0.5), pyautogui.click())
LURE_EQUIP = lambda: (pyautogui.moveTo(1650, 800, duration=0.5), pyautogui.rightClick())
RESET_MOUSE = lambda: (pyautogui.moveTo(1920 / 2, 1080 / 2, duration=0.5))

GODI_PATH='Y:\\dbc\\GameObjectDisplayInfo.dbc'
CMD_PATH='Y:\\dbc\\CreatureModelData.dbc'
MODELS_PREFIX='Y:\\model.mpq'
w = WoW(godi_path=GODI_PATH, cmd_path=CMD_PATH)

class S(Enum):
    BOBBER_FLYING = 1
    BOBBER_TARGETED = 2
    BOBBER_DISAPPEARING = 3

def main():
    s = S.BOBBER_DISAPPEARING
    go = None
    idx = 0
    black_list_model_matrices = []
    t0 = datetime.datetime.now() - datetime.timedelta(minutes=20)

    def panic(msg):
        nonlocal s
        s = S.BOBBER_DISAPPEARING
        print(msg)
        print()
        time.sleep(1.6)
        STOP()
        time.sleep(1.6)

    panic("> prime")
    while True:
        if s == S.BOBBER_DISAPPEARING:
            print("> Player should be ready to cast")
            black_list_model_matrices = [
                tuple(go.model_matrix.flatten().tolist())
                for go in w.gen_game_objects()
                if go.name == "Flotteur"
                for xyz in [(0, 0, 0, 1) @ go.model_matrix]
            ]
            now = datetime.datetime.now()
            if LURE and (now - t0).total_seconds() > 60 * 9:
                print("| applying lure")

                LURE_SELECT()
                LURE_APPLY()
                LURE_CONFIRM()
                time.sleep(5)
                LURE_EQUIP()
                RESET_MOUSE()

                t0 = now
                continue

            idx = 0
            print("| middle clicking and zZz")
            FISH()
            s = S.BOBBER_FLYING
            time.sleep(1)
        elif s == S.BOBBER_FLYING:
            print("> Bobber should have landed")
            cam = Camera(w)
            gos = [
                (go, x, y)
                for go in w.gen_game_objects()
                if go.name == "Flotteur"
                for xyz in [(0, 0, 0, 1) @ go.model_matrix]
                for in_bl in [tuple(go.model_matrix.flatten().tolist()) in black_list_model_matrices]
                for (x, y), visible, behind in [cam.world_to_screen(xyz[:3])]
                for _ in [print(f"| bobber: guid:{go.guid} x:{x} y:{y} visible:{visible} behind:{behind} in_bl:{in_bl}")]
                if visible and not behind and not in_bl
            ]
            if len(gos) != 1:
                panic(f"| {len(gos)} bobbers found")
                continue
            go, x, y = gos[0]
            print("| moving to bobber and zZz")
            pyautogui.moveTo(x, y)
            time.sleep(0.5)
            s = S.BOBBER_TARGETED
        elif s == S.BOBBER_TARGETED:
            if idx % CPS == 0:
                print(f"> Should be targeted (rep={idx})")
            idx += 1
            if idx > CPS * 26:
                panic("| fade")
                continue
            guid = int(w.pull_u32s(Offset.mouseover_guid, ()))
            status = int(w.pull_u32s(go.addr + Offset.Object.bobber_status, ()))
            if guid != go.guid:
                panic(f"| Mouse is over guid={guid} (need {go.guid})")
                continue
            if status != 1:
                if idx % CPS == 0:
                    print(f"|  status={status} now ZzZ")
                time.sleep(1. / CPS)
                continue
            print("? shift click")
            pyautogui.keyDown('shift')
            pyautogui.rightClick()
            pyautogui.keyUp('shift')
            s = S.BOBBER_DISAPPEARING
            time.sleep(1.6)
            print()
        else:
            assert False

main()
