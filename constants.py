
"""
widths = [i for i in range(64, 1921, 64)]
combos = list(itertools.product(widths, widths))
combos = sorted(combos, key=lambda t: -abs(t[0] / t[1] - 16/9))
combos = [(x, y) for (x, y) in combos if -0.20 < x/y / (16/9) - 1 < 0.20]

['{:9}, ratio {:+.2%}, memsize {:4.0%}'.format('{}/{}'.format(x, y), x/y / (16/9) - 1, x*y / (1920*1080)) for (x, y) in combos]

 '448/256  , ratio -1.56%, memsize   6%',
 '896/512  , ratio -1.56%, memsize  22%',
 '1344/768 , ratio -1.56%, memsize  50%',
 '1792/1024, ratio -1.56%, memsize  88%',
 '576/320  , ratio +1.25%, memsize   9%',
 '1152/640 , ratio +1.25%, memsize  36%',
 '1728/960 , ratio +1.25%, memsize  80%',
 '1920/1088, ratio -0.74%, memsize 101%',
 '1472/832 , ratio -0.48%, memsize  59%',
 '1600/896 , ratio +0.45%, memsize  69%',
 '1024/576 , ratio +0.00%, memsize  28%']
"""

# img_w = 64
# img_h = 64
img_w = 1024
img_h = 576
# img_w = 576
# img_h = 320
img_d = 3

n_labels = 2

kernel = 3

prefix = 'C:/Users/Ngo/Desktop/fishdb/current'
# prefix = '/media/ngoguey/Donnees/ngoguey/fishbd'

time_format = '%y-%m-%d-%H-%M-%S'
