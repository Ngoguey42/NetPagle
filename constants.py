
MAGIC_SCALE_FACTOR = 1 / 1.096

# *********************************************************************************************** **
class Offset:
    player_name = 0x827D88
    obj_manager = 0x00741414
    camera = 0x0074B2BC

    class PlayerNameCache:
        # Cycling linked lst
        root = 0xC0E230
        class Entry:
            next_addr = 0x0
            guid = 0xc
            name = 0x14

    class ObjectManager:
        first_obj = 0xAC

    class Object:
        type = 0x14
        next = 0x3C
        guid = 0x30

    class GameObject:
        guid = 0x30
        name1 = 0x214
        name2 = 0x8
        xyz = 0x2c4
        angle = xyz + 3 * 4
        quaternion = xyz - 5 * 4
        display_id = 0x2a8
        unknown_matrix = 0x218
        scale = 0x298

    class Camera:
        offset = 0x65B8
        xyz = 0x8
        facing = xyz + 3 * 4
        fov = xyz + 14 * 4
        aspect = xyz + 15 * 4

    class Player:
        xyz = 0x9b8
        angle = 0x9c4
        display_ids = 0x1f7c
        level = 0x1df8
        unitflags = 0x1e00

# *********************************************************************************************** **
race_name_of_race_id = {
    0: 'None',
    1: 'Human',
    2: 'Orc',
    3: 'Dwarf',
    4: 'Night Elf',
    5: 'Undead',
    6: 'Tauren',
    7: 'Gnome',
    8: 'Troll',
}
race_id_of_race_name = {v: k for k, v in race_name_of_race_id.items()}

class_name_of_class_id = {
    0: 'None',
    1: 'Warrior',
    2: 'Paladin',
    3: 'Hunter',
    4: 'Rogue',
    5: 'Priest',
    7: 'Shaman',
    8: 'Mage',
    9: 'Warlock',
    11: 'Druid',
}
class_id_of_class_name = {v: k for k, v in class_name_of_class_id.items()}

gender_name_of_gender_id = {
    0: 'Male',
    1: 'Female',
}
gender_id_of_gender_name = {v: k for k, v in gender_name_of_gender_id.items()}

# *********************************************************************************************** **
def set_pretty_print_env(level=None):
    import logging
    import numpy as np
    import pandas as pd
    import warnings

    np.set_printoptions(linewidth=250, threshold=999999999999, suppress=True)

    if level is None:
        level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)

    pd.set_option('display.width', 260)
    pd.set_option('display.max_colwidth', 260)
    pd.set_option('display.float_format', lambda x: '%.8f' % x)
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.max_rows', 210)
    # pd.set_option('display.max_rows', 125)

    # http://stackoverflow.com/a/7995762
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.INFO, "\033[1;34m%s \033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.DEBUG, "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))

    logging.getLogger('matplotlib').setLevel(logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

set_pretty_print_env()
del set_pretty_print_env
