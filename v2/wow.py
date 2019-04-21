import numpy as np
import pymem
import psutil

from constants import Offset
from dbc import GameObjectDisplayInfo
from objects import GameObject

class WoW:
    def __init__(self, pid=None, godi_path='Y:\\dbc\\GameObjectDisplayInfo.dbc'):
        if pid is None:
            pid, = [ps.pid for ps in psutil.process_iter() if ps.name() == 'WoW.exe']

        pm = pymem.Pymem()
        pm.open_process_from_id(pid)
        self.pid = pid
        self.pm = pm

        mod, = [mod for mod in pm.list_modules() if mod.name == 'WoW.exe']
        base_address = mod.lpBaseOfDll
        obj_mgr_addr = pm.read_uint(base_address + Offset.obj_manager)
        first_obj_addr = pm.read_uint(obj_mgr_addr + Offset.ObjectManager.first_obj)

        self.mod = mod
        self.base_address = base_address
        self.player_name = pm.read_string(self.base_address + Offset.player_name)
        self.first_obj_addr = first_obj_addr

        self.godi = GameObjectDisplayInfo(godi_path)

    def pull_floats(self, addr, shape):
        a = np.empty(shape, 'float32')
        for i, idxs in enumerate(np.ndindex(*shape)):
            a[idxs] = self.pm.read_float(addr + i * 4)
        return a.astype('float64')

    def pull_u32s(self, addr, shape):
        a = np.empty(shape, 'uint32')
        for i, idxs in enumerate(np.ndindex(*shape)):
            a[idxs] = self.pm.read_uint(addr + i * 4)
        return a

    def gen_objects(self):
        obj_addr = self.first_obj_addr
        while True:
            if not (obj_addr != 0 and obj_addr & 0x1 == 0):
                break
            yield obj_addr
            next_addr = self.pm.read_uint(obj_addr + Offset.Object.next)
            if next_addr == obj_addr:
                break
            obj_addr = next_addr

    def gen_game_objects(self):
        for obj_addr in self.gen_objects():
            if self.pm.read_int(obj_addr + Offset.Object.type) == 5:
                yield GameObject(self, obj_addr)
