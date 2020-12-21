# only runs through terminal. IPython console not supported

import numpy as np
import os
import time
import math
import ffmpeg
from scipy import interpolate
from matplotlib import pyplot as plt
from skimage import util, filters
import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'


def print_title(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def imshow(src, pts=None, line=False, width=1.5, path=None):
    plt.figure(dpi=200)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(src)
    if pts is not None:
        pts = np.array(pts)
        pts = np.concatenate((pts, pts[0][None, :]), axis=0)
        pts = pts.T.astype(int)
        if line:
            plt.plot(pts[1], pts[0], '-', lw=width)
        plt.plot(pts[1], pts[0], '.r', markersize=1.5)

    if path is None:
        # plt.show()
        pass
    else:
        plt.savefig(path, dpi=200)
        plt.close('all')


img_o = plt.imread('./data/tasbih.jpg')
n = 100

plt.clf()
plt.imshow(img_o)

init = []

while True:
    while len(init) < 3:
        print_title('Select at least 3 corners, press \'esc\' to continue')
        init = np.asarray(plt.ginput(-1, timeout=-1))
        if len(init) < 3:
            print_title('Too few points, starting over')
            time.sleep(0.2)  # Wait a second
    init = np.concatenate((init, init[0][None, :]), axis=0)
    ph = plt.plot(init[:, 0], init[:, 1], 'r', lw=1)
    print_title('Happy? \'esc\' for yes, mouse click for no')
    if plt.waitforbuttonpress():
        break
    for p in ph:
        p.remove()
plt.close()

init = init[:, ::-1]

tck, *u = interpolate.splprep(init.T, s=0, k=1)
unew = np.linspace(0, 1, n)
init = np.array(interpolate.splev(unew, tck)).T

plt.clf()
plt.imshow(img_o)
temp = np.concatenate((init, init[0][None, :]), axis=0)
plt.plot(temp[:, 1], temp[:, 0], 'g')

print_title('Click to begin process...')
plt.waitforbuttonpress()
plt.close()

time.sleep(0.5)

# start of process

img = util.img_as_float64(img_o, True)
img = filters.gaussian(img, 3, multichannel=True)

img_gr = np.zeros(img.shape[:2])
for i in range(3):
    gry = filters.sobel_h(img[..., i])
    grx = filters.sobel_v(img[..., i])
    img_gr += np.sqrt(grx ** 2 + gry ** 2)

thr = filters.threshold_otsu(img_gr)
img_gr *= img_gr > thr
img_gr = filters.gaussian(img_gr, 7)
img_gr /= img_gr.max(initial=1e-12)


def sigmoid(x):
    return 2 / (1 + math.exp(-20 * x)) - 1


px_mv = 2
max_iter = 180
frm = 2  # capture framerate

snake = init.copy()
dpx = 2 * px_mv + 1
sqpx = dpx ** 2

dir_name = str(time.time())
os.makedirs('./out/' + dir_name)

# e_ext = - \gamma img_gr[i,j]
# e_int = \alpha \sum(|v_{i} - v_{i-1}|^2 - d)^2 + \beta \sum |v_i-v|

for iter_num in range(max_iter):
    dp = np.zeros((n, sqpx))  # each entry shows energy
    dp_ref = np.zeros_like(dp, dtype=int)  # each entry shows next

    cntr = np.mean(snake, axis=0)

    alpha = 0.03 / np.linalg.norm(img.shape)
    beta = .50 / np.linalg.norm(img.shape) / n
    gamma = 1000 / n

    drv_avg = snake - np.roll(snake, 1, axis=0)
    drv_avg = np.sqrt(drv_avg[:, 0] ** 2 + drv_avg[:, 1] ** 2)
    drv_avg = np.mean(drv_avg, axis=0)

    dist_cnt_avg = snake - cntr[None, :]
    dist_cnt_avg = np.sqrt(dist_cnt_avg[:, 0] ** 2 + dist_cnt_avg[:, 1] ** 2)
    dist_cnt_avg = np.mean(dist_cnt_avg, axis=0)

    # dp optimization
    for k in range(n):
        for t1 in range(sqpx):
            cur_pt = snake[k] + (t1 / dpx - px_mv, t1 % dpx - px_mv)
            e_ext = -gamma * img_gr[tuple(cur_pt.astype(int))]
            mn = np.inf
            arg_mn = 0
            for t2 in range(sqpx):
                prev_pt = snake[k - 1] + (t2 / dpx - px_mv, t2 % dpx - px_mv)
                val = dp[k - 1, t2]
                val += alpha * (np.linalg.norm(cur_pt - prev_pt) ** 2 - drv_avg) ** 2
                if val < mn:
                    mn = val
                    arg_mn = t2
            mn += beta * (np.linalg.norm(cur_pt - cntr) ** 2 - dist_cnt_avg / 2) ** 2 * (
                    1 - sigmoid(img_gr[tuple(cur_pt.astype(int))]))
            mn += e_ext
            dp[k, t1] = mn
            dp_ref[k, t1] = arg_mn

    # update snake
    mn = np.inf
    arg_mn = 0
    for t in range(sqpx):
        if dp[n - 1, t] < mn:
            mn = dp[n - 1, t]
            arg_mn = t
    tmp = snake[n - 1] + (arg_mn / dpx - px_mv, arg_mn % dpx - px_mv)
    for k in range(n - 1):
        arg_mn = dp_ref[n - k - 1, arg_mn]
        snake[n - k - 1] = snake[n - k - 2] + (arg_mn / dpx - px_mv, arg_mn % dpx - px_mv)
    snake[0] = tmp

    print(iter_num)
    if iter_num % frm == 0:
        imshow(img_o, snake, True, path=f'./out/{dir_name}/{iter_num:03d}.jpg')

(
    ffmpeg
        .input(f'./out/{dir_name}/*.jpg', pattern_type='glob', framerate=10 / frm)
        .output(f'./out/{dir_name}/contour.mp4', crf=10)
        .run()
)
imshow(img_o, snake, True, path=f'./out/{dir_name}/res10.jpg')
