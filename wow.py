import itertools

import numpy as np
import pymem
import psutil
import pandas as pd

from constants import Offset
from dbc import GameObjectDisplayInfo, CreatureModelData
from objects import GameObject, Player

class WoW:
    def __init__(self, pid=None, godi_path=None, cmd_path=None):
        if pid is None:
            pids = [ps.pid for ps in psutil.process_iter() if ps.name() == 'WoW.exe']
            if len(pids) <= 1:
                pid, = pids
            else:
                for i, pid in enumerate(pids):
                    print(i, pid)
                pid = pids[int(input().strip())]

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
        self.cmd = CreatureModelData(cmd_path)
        self._player_name_per_guid = {}

    def get_player_name(self, req_guid):
        req_guid = int(req_guid)
        if req_guid not in self._player_name_per_guid:
            addr = self.pull_u32s(Offset.PlayerNameCache.root)
            while True:
                if addr == Offset.PlayerNameCache.root:
                    break
                guid = int(self.pull_u64s(addr + Offset.PlayerNameCache.Entry.guid))
                if guid not in self._player_name_per_guid:
                    name = self.pull_cstring(addr + Offset.PlayerNameCache.Entry.name)
                    self._player_name_per_guid[guid] = name
                addr = self.pull_u32s(addr + Offset.PlayerNameCache.Entry.next_addr)

        return self._player_name_per_guid[req_guid]

    # ******************************************************************************************* **
    def pull_cstring(self, addr):
        addr = int(addr)
        s = b''
        while True:
            t = self.pm.read_bytes(addr, 1)
            if t == b'\0':
                break
            s += t
            addr += 1
        s = s.decode('utf8')
        return s

    def pull_floats(self, addr, shape=()):
        addr = int(addr)
        a = np.empty(shape, 'float32')
        for i, idxs in enumerate(np.ndindex(*shape)):
            a[idxs] = self.pm.read_float(addr + i * 4)
        return a.astype('float64')

    def pull_u32s(self, addr, shape=()):
        addr = int(addr)
        a = np.empty(shape, 'uint32')
        for i, idxs in enumerate(np.ndindex(*shape)):
            a[idxs] = self.pm.read_uint(addr + i * 4)
        return a

    def pull_u64s(self, addr, shape=()):
        addr = int(addr)
        a = np.empty(shape, 'uint64')
        for i, idxs in enumerate(np.ndindex(*shape)):
            a[idxs] = self.pm.read_uint(addr + i *8)
        return a

    def view(self, addr, byte_count):
        addr = int(addr)
        byte_count = int(byte_count)
        bts = self.pm.read_bytes(addr, byte_count)
        return pd.Series({
            'addr': addr,
            'byte_count': byte_count,
            'bts': bts,
            'asascii': ''.join(
                chr(b) if (31 < b < 127) else '?'
                for b in bts
            ),
            'asu8': np.frombuffer(bts, 'uint8'),
            'asu16': np.frombuffer(bts[:len(bts) // 2 * 2], 'uint16'),
            'asu32': np.frombuffer(bts[:len(bts) // 4 * 4], 'uint32'),
            'asfloat': np.frombuffer(bts[:len(bts) // 4 * 4], 'float32'),
        })

    def gen_objects_addr(self):
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
        for obj_addr in self.gen_objects_addr():
            if self.pm.read_int(obj_addr + Offset.Object.type) == 5:
                yield GameObject(self, obj_addr)

    def gen_players(self):
        for obj_addr in self.gen_objects_addr():
            if self.pm.read_int(obj_addr + Offset.Object.type) == 4:
                yield Player(self, obj_addr)
