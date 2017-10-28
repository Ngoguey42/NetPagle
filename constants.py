import os

"""
widths = [i for i in range(64, 1921, 64)]
combos = list(itertools.product(widths, widths))
combos = sorted(combos, key=lambda t: -abs(t[0] / t[1] - 16/9))
combos = [(x, y) for (x, y) in combos if -0.20 < x/y / (16/9) - 1 < 0.20]

['{:9}, ratio {:+.2%}, memsize {:4.0%}'.format('{}/{}'.format(x, y), x/y / (16/9) - 1, x*y / (1920*1080)) for (x, y) in combos]


 '832/512  , ratio -8.59%, memsize  21%',
 '1152/704 , ratio -7.95%, memsize  39%',
 '1216/640 , ratio +6.88%, memsize  38%',
 '1088/576 , ratio +6.25%, memsize  30%',
 '320/192  , ratio -6.25%, memsize   3%',
 '640/384  , ratio -6.25%, memsize  12%',
 '960/576  , ratio -6.25%, memsize  27%',
 '960/512  , ratio +5.47%, memsize  24%',
 '832/448  , ratio +4.46%, memsize  18%',
 '1088/640 , ratio -4.37%, memsize  34%',
 '768/448  , ratio -3.57%, memsize  17%',
 '704/384  , ratio +3.12%, memsize  13%',
 '448/256  , ratio -1.56%, memsize   6%',
 '896/512  , ratio -1.56%, memsize  22%',
 '576/320  , ratio +1.25%, memsize   9%',
 '1152/640 , ratio +1.25%, memsize  36%',
 '1920/1088, ratio -0.74%, memsize 101%',
 '1024/576 , ratio +0.00%, memsize  28%']
"""

# img_w = 64
# img_h = 64
img_w = 1024
img_h = 576
# img_w = 576
# img_h = 320
img_d = 3

n_labels = 1

kernel = 3

prefix = 'C:/Users/Ngo/Desktop/fishdb/current'
# prefix = '/media/ngoguey/Donnees/ngoguey/fishbd'

info_path = os.path.join(prefix, 'models', 'info.yml')

time_format = '%y-%m-%d-%H-%M-%S'
