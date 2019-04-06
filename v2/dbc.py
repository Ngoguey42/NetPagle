import numpy as np
import pandas as pd

class GameObjectDisplayInfo:
    def __init__(self, path):
        self.bts = open(path, 'rb').read()

        print(self.bts[:100])
        assert self.magic == b'WDBC'
        print(self.pull_u32s(0x4, 4))

        print(len(self.bts))
        assert len(self.bts) == (
            5 * 4 + self.string_block_size + self.record_count * self.record_size
        )
        rows = []
        for addr in self.record_addresses:
            id_, str_addr = self.pull_u32s(addr, 2)
            str_addr += self.string_block_address
            model = self.cstring(str_addr)
            rows.append({
                'id': id_,
                'model': model,
            })
        self.df = pd.DataFrame(rows).set_index('id', drop=1).sort_index()

    # Offsets *********************************************************************************** **
    @property
    def magic(self):
        return self.bts[:4]

    @property
    def record_count(self):
        return self.pull_u32s(4 * 1)

    @property
    def field_count(self):
        return self.pull_u32s(4 * 2)

    @property
    def record_size(self):
        return self.pull_u32s(4 * 3)

    @property
    def string_block_size(self):
        return self.pull_u32s(4 * 4)

    @property
    def record_addresses(self):
        return 4 * 5 + np.arange(self.record_count) * self.record_size

    @property
    def string_block_address(self):
        return 5 * 4 + self.record_count * self.record_size

    # ******************************************************************************************* **
    def cstring(self, addr, sep=0):
        """m2array<char>"""
        i = self.bts.find(sep, addr)
        n = self.bts[addr:i]
        assert i != 1
        n = n.decode('utf8')
        assert n.isprintable(), (n, addr, sep, i)
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
    path = 'Y:\\dbc\\GameObjectDisplayInfo.dbc'
    godi = GameObjectDisplayInfo(path)
    df = godi.df
