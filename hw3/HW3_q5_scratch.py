# %%
import numpy as np
import cv2 as cv
import os
import time
import math
import ffmpeg
from scipy import interpolate
from matplotlib import pyplot as plt
from skimage import util, exposure, filters, restoration, segmentation

img_o = plt.imread('./data/tasbih.jpg')

n = 100
s = np.linspace(0, 2 * np.pi, n)
r = 400 + 300 * np.sin(s)
c = 450 + 250 * np.cos(s)  # 580 400
init = np.array([r, c]).T


def imshow(src, pts, line=False, width=1.5, path=None):
    src = src.copy()
    pts = pts.T.astype(int)

    plt.figure(dpi=200)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(src)
    if line:
        plt.plot(pts[1], pts[0], '-', lw=width)
    plt.plot(pts[1], pts[0], '.r', markersize=1.5)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=200)
        plt.close('all')


imshow(img_o, init)
# %%
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

plt.imshow(img_gr)
plt.show()


# %%

def sigmoid(x):
    return 2 / (1 + math.exp(-20 * x)) - 1


px_mv = 2
max_iter = 180
frm = 2

snake = init.copy()
dpx = 2 * px_mv + 1
sqpx = dpx ** 2

dir_name = str(time.time())
os.makedirs('./video/' + dir_name)

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
        imshow(img_o, snake, True, path=f'./video/{dir_name}/{iter_num:03d}.jpg')

imshow(img_o, snake)
(
    ffmpeg
        .input(f'./video/{dir_name}/*.jpg', pattern_type='glob', framerate=10 / frm)
        .output(f'./video/{dir_name}/contour.mp4', crf=10)
        .run()
)
# %%
imshow(img_o, snake, True, path=f'./video/{dir_name}/res09.jpg')
