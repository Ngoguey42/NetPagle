
# MAGIC_SCALE_FACTOR = 1 / 1.029 # TODO: Find the real formula
MAGIC_SCALE_FACTOR = 1 / 1.096 # TODO: Find the real formula

# SCREEN_SIZE = 1920, 1080 # TODO: Find in memory

def set_pretty_print_env(level=None):
    import logging
    import numpy as np
    import pandas as pd
    import warnings

    np.set_printoptions(linewidth=250, threshold=np.nan, suppress=True)

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

class Offset:
    player_name = 0x827D88
    obj_manager = 0x00741414
    camera = 0x0074B2BC

    class ObjectManager:
        first_obj = 0xAC

    class Object:
        type = 0x14
        next = 0x3C

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
