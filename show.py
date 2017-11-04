import sys
import os

import model
import data_source
from constants import *

ds = data_source.DataSource(PREFIX)
m = model.Model(os.path.join(PREFIX, sys.argv[1]), ds)
m.show_board()
