# %%
import numpy as np
from skimage import util
from skimage import draw
import cv2 as cv
from matplotlib import pyplot as plt
import networkx as nx
import collections
import math


def imshow(*imgs):
    plt.close()
    fig, axs = plt.subplots(ncols=len(imgs))
    if len(imgs) == 1:
        plt.imshow(imgs[0])
    else:
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
    plt.show()


texture_o = cv.cvtColor(cv.imread('./data/texture2.jpg'), cv.COLOR_BGR2RGB)
texture_g = cv.cvtColor(texture_o, cv.COLOR_RGB2GRAY)
texture_f = util.img_as_float64(texture_o)
plt.imshow(texture_o)
plt.show()

sz_tar_f = np.array([2500, 2500])
sz_tex = np.array(texture_o.shape[:2])
sz_smp = 300
sz_ovlp = 100
cnt_smp = np.ceil(np.array(sz_tar_f) / sz_smp).astype(np.int)
sz_tar = cnt_smp * sz_smp


# %%
def draw_sample(src, sample_size):
    pt = np.random.randint(0, high=np.array(src.shape[:2]) - sample_size)
    return src[pt[0]:pt[0] + sz_smp, pt[1]:pt[1] + sz_smp]


def draw_similar_sample(src, templ, fld):
    mtc = cv.matchTemplate(fld, templ, cv.TM_CCOEFF_NORMED)
    imshow(mtc)
    print(mtc.max(), mtc.std(), mtc.mean())

    # aw = np.argwhere(mtc > mtc.mean() + 2 * mtc.std())
    # print(aw.shape)
    # pt = aw[np.random.randint(aw.shape[0], size=1)][0]
    pt = np.unravel_index(np.argmax(mtc), mtc.shape)
    print(pt)
    return src[pt[0]:pt[0] + templ.shape[0], pt[1]:pt[1] + templ.shape[1]]


def get_mag(src):
    src = src ** 2
    res = np.zeros(src.shape[:2])
    if len(src.shape) > 2:
        for i in range(3):
            res += src[:, :, i]
    else:
        res += src
    res = np.sqrt(res)
    return res


tar = np.zeros(sz_tar, dtype=np.float64)

smp1 = draw_sample(texture_o, sz_smp)
smp2 = draw_similar_sample(texture_o[::-1, ::-1, :], cv.cvtColor(smp1, cv.COLOR_RGB2GRAY), texture_g[::-1, ::-1])

smp = get_mag(smp1 - smp2)

imshow(smp1, smp2, smp)

# %%
smpp = cv.GaussianBlur(smp, (5, 5), 0)


def find_shortest_path(mat):
    G = nx.DiGraph()
    for j in range(mat.shape[1]):
        G.add_edge(-1, j, weight=mat[0, j])
        G.add_edge(mat.size - 1 - j, -2, weight=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i > 0:
                G.add_edge(i * mat.shape[1] + j, (i - 1) * mat.shape[1] + j, weight=mat[i - 1, j])
                if j > 0:
                    G.add_edge(i * mat.shape[1] + j, (i - 1) * mat.shape[1] + j - 1, weight=mat[i - 1, j - 1])
            if j > 0:
                G.add_edge(i * mat.shape[1] + j, i * mat.shape[1] + j - 1, weight=mat[i, j - 1])
                if i < mat.shape[0] - 1:
                    G.add_edge(i * mat.shape[1] + j, (i + 1) * mat.shape[1] + j - 1, weight=mat[i + 1, j - 1])
            if i < mat.shape[0] - 1:
                G.add_edge(i * mat.shape[1] + j, (i + 1) * mat.shape[1] + j, weight=mat[i + 1, j])
                if j < mat.shape[1] - 1:
                    G.add_edge(i * mat.shape[1] + j, (i + 1) * mat.shape[1] + j + 1, weight=mat[i + 1, j + 1])
            if j < mat.shape[1] - 1:
                G.add_edge(i * mat.shape[1] + j, i * mat.shape[1] + j + 1, weight=mat[i, j + 1])
                if i > 0:
                    G.add_edge(i * mat.shape[1] + j, (i - 1) * mat.shape[1] + j + 1, weight=mat[i - 1, j + 1])
    path = nx.shortest_path(G, source=-1, target=-2, weight='weight')
    res = []
    for vert in path[1:-1]:
        res.append((vert // mat.shape[1], vert % mat.shape[1]))
    return np.array(res)


def find_shortest_path_dp(mat, spr=1, diag=1):
    dp = np.ones_like(mat) * 1000000000
    dp_arg = np.ones(mat.shape, dtype=int) * -1
    dp[0, :] = 0
    dp_arg[0, :] = -1
    for i in range(1, mat.shape[0]):
        for j in range(mat.shape[1]):
            for j2 in range(max(0, j - spr), min(mat.shape[1], j + spr + 1)):
                dist = 0
                if (abs(j - j2) > diag):
                    pass
                else:
                    dist = dp[i - 1, j2] + mat[i - 1, j2]
                if dist < dp[i, j]:
                    dp[i, j] = dist
                    dp_arg[i, j] = j2
    arg_mn = np.argmin(dp[-1, :])
    arg_i = mat.shape[0] - 1
    res = []
    res.append((arg_i, arg_mn))
    while arg_i > 0:
        arg_mn = dp_arg[arg_i, arg_mn]
        arg_i -= 1
        res.append((arg_i, arg_mn))
    res.reverse()
    return np.array(res)


fig, axs = plt.subplots(ncols=3, figsize=(30, 10))
[axi.set_axis_off() for axi in axs.ravel()]
tt = find_shortest_path(smpp)
axs[0].imshow(smp)
axs[0].plot(tt[:, 1], tt[:, 0], lw=1, c='r')
tt2 = find_shortest_path_dp(smpp)
axs[1].imshow(smp)
axs[1].plot(tt2[:, 1], tt2[:, 0], lw=1, c='r')
axs[2].imshow(smpp)
plt.show()

# %%
tt3 = np.concatenate((tt2, [[smp.shape[0], 0], [0, 0]]), axis=0)
mask = draw.polygon2mask(smp.shape[:2], tt3) * np.ones_like(smp)
mask = cv.GaussianBlur(mask, (11, 11), 0)
# plt.imshow(mask)
# plt.show()
# %%
fig, axs = plt.subplots(ncols=3, figsize=(30, 10))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0].imshow(smp1)
axs[1].imshow(smp2)

smp_fin = mask[:, :, None] * smp1 + (1 - mask)[:, :, None] * smp2
smp_fin = smp_fin.astype(np.uint8)
axs[2].imshow(smp_fin)
plt.show()
