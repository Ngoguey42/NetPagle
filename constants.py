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

PREFIX = 'C:/Users/Ngo/Desktop/fishdb/current'
# PREFIX = '/media/ngoguey/Donnees/ngoguey/fishbd'

time_format = '%y-%m-%d-%H-%M-%S'

TEST_NAMES = [
	'17-10-28-21-24-48_red-stonetalon-sunrock-scroll10_anna',
	'17-10-28-21-25-29_red-stonetalon-sunrock-scroll10_shirley',
	'17-10-28-21-26-08_red-stonetalon-sunrock-scroll10_june',
	'17-10-28-21-27-35_red-stonetalon-sunrock-scroll10_clara-occlusion',
	'17-10-28-21-29-13_red-stonetalon-sunrock-scroll10_jean',
	'17-10-24-23-22-32_blue-darnassus-auctionhouse-scroll0_mildred',
	'17-10-24-23-24-24_blue-darnassus-auctionhouse-scroll0_brett',
	'17-10-24-23-26-01_blue-darnassus-auctionhouse-scroll0_gail',
	'17-10-24-23-27-13_blue-darnassus-auctionhouse-scroll0_mary',
	'17-10-24-23-28-52_blue-darnassus-auctionhouse-scroll0_gina',
	'17-10-28-21-09-56_green-barrens-stagnantoasis-scroll10_lisa',
	'17-10-28-20-09-45_black-silverpine-bridge-scroll5_amparo',
	'17-10-28-20-30-27_black-hillsbrad-cascade-scroll5_richard',
	'17-10-28-20-16-01_black-hillsbrad-tarren-scroll5_robert',
	'17-10-28-19-55-15_black-silverpine-lake-scroll5_norma-occlusion',
	'17-10-27-00-39-54_green-moonglade-river-scroll0_leann',
	'17-10-27-08-56-45_blue-moonglade-tree-scroll0_raymond',
	'17-10-24-22-58-40_blue-strangle-bootybay-scroll0_samantha',
	'17-10-24-23-02-32_green-thunderbluff-poolsofvision-scroll0_alan',
	'17-10-24-22-45-35_red-orgrimmar-valleyofhonor-scroll0_delores',
]
