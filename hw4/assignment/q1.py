import numpy as np
from skimage import draw
import cv2 as cv
from matplotlib import pyplot as plt

texture_o = cv.cvtColor(cv.imread('./data/texture1.jpg'), cv.COLOR_BGR2RGB)

sz_tar_f = np.array([2500, 2500])
sz_smp = 300
sz_ovlp = 100

cnt_smp = np.ceil(np.array(sz_tar_f) / (sz_smp - sz_ovlp)).astype(np.int)
sz_tar = cnt_smp * (sz_smp - sz_ovlp) + sz_ovlp

texture_g = []
texture = []
texture_g.append(cv.cvtColor(texture_o, cv.COLOR_RGB2GRAY))
texture.append(texture_o.copy())
for i in range(3):
    texture_g.append(np.transpose(texture_g[-1][::-1, :]))
    texture.append(np.transpose(texture[-1][::-1, :, :], axes=[1, 0, 2]))
texture_g.append(np.transpose(cv.cvtColor(texture_o, cv.COLOR_RGB2GRAY)))
texture.append(np.transpose(texture_o.copy(), axes=[1, 0, 2]))
for i in range(3):
    texture_g.append(np.transpose(texture_g[-1][::-1, :]))
    texture.append(np.transpose(texture[-1][::-1, :, :], axes=[1, 0, 2]))


# %%

def draw_sample(src, pt, size):
    size = np.array(size)
    if size.size < 2:
        size = np.array([size, size])
    return src[pt[0]:pt[0] + size[0], pt[1]:pt[1] + size[1]]


def draw_random_sample(src, size):
    size = np.array(size)
    if size.size < 2:
        size = np.array([size, size])
    rect = np.array(src.shape[:2]) - size
    pt = np.random.randint(0, high=rect)
    return draw_sample(src, pt, size)


def find_similar_sample(fld, templ, mask=None):
    mtc = cv.matchTemplate(fld, templ, cv.TM_CCORR_NORMED, mask=mask)
    return np.unravel_index(np.argmax(mtc), mtc.shape)


def draw_random_similar_sample(srcs, templ, flds, l_strip, hor=False, ver=True):
    """template and field must be in same color space."""
    size = templ.shape
    kt = np.random.randint(0, len(srcs))
    src = srcs[kt]
    fld = flds[kt]
    mask = np.zeros(templ.shape, dtype=np.uint8)
    a = b = -1
    if hor:
        mask[:l_strip, :] = 255
        a = -size[0] + l_strip
    if ver:
        mask[:, :l_strip] = 255
        b = -size[1] + l_strip
    pt = find_similar_sample(fld[:a, :b], templ, mask)
    return draw_sample(src, pt, size)


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


def find_shortest_path(mat, spr=1, diag=1):
    dp = np.ones_like(mat) * 1000000000
    dp_arg = np.ones(mat.shape, dtype=int) * -1
    dp[0, :] = 0
    dp_arg[0, :] = -1
    for i in range(1, mat.shape[0]):
        for j in range(mat.shape[1]):
            for j2 in range(max(0, j - spr), min(mat.shape[1], j + spr + 1)):
                dist = 0
                if abs(j - j2) > diag:
                    pass
                else:
                    dist = dp[i - 1, j2] + mat[i - 1, j2]
                if dist < dp[i, j]:
                    dp[i, j] = dist
                    dp_arg[i, j] = j2
    arg_mn = np.argmin(dp[-1, :])
    arg_i = mat.shape[0] - 1
    res = [(arg_i, arg_mn)]
    while arg_i > 0:
        arg_mn = dp_arg[arg_i, arg_mn]
        arg_i -= 1
        res.append((arg_i, arg_mn))
    res.reverse()
    return np.array(res)


def get_shortest_path_mask(mat, spr=1, diag=1, ker_prep=5):
    mat = cv.GaussianBlur(mat, (ker_prep, ker_prep), 0)
    verts = find_shortest_path(mat, spr, diag)
    verts = np.concatenate((verts, [[mat.shape[0], 0], [0, 0]]), axis=0)
    return draw.polygon2mask(mat.shape[:2], verts)


def get_mask_cut(tmpl1, tmpl2, l_strip, hor=False, ver=True, ker_edge=15, ker_prep=5):
    mask = np.zeros(tmpl2.shape[:2], dtype=np.bool)
    if ver:
        mask[:, :l_strip] |= get_shortest_path_mask(get_mag(tmpl1[:, :l_strip] - tmpl2[:, :l_strip]),
                                                    ker_prep=ker_prep)
    if hor:
        mask[:l_strip, :] |= get_shortest_path_mask(get_mag(tmpl1[:l_strip, :] - tmpl2[:l_strip, :]).T,
                                                    ker_prep=ker_prep).T
    mask = mask * 1.0
    mask = cv.GaussianBlur(mask, (ker_edge, ker_edge), 0)
    return mask


res = np.zeros((*sz_tar, 3), dtype=np.uint8)

for i, j in np.ndindex(*cnt_smp):
    if i == 0:
        if j == 0:
            res[:sz_smp, :sz_smp] = draw_random_sample(texture[0], sz_smp)
            continue
    r = i * (sz_smp - sz_ovlp)
    c = j * (sz_smp - sz_ovlp)
    smp_old = res[r:r + sz_smp, c:c + sz_smp]
    smp_new = draw_random_similar_sample(texture, smp_old, texture, sz_ovlp
                                         , i != 0, j != 0)
    msk = get_mask_cut(smp_old, smp_new, sz_ovlp, i != 0, j != 0)[:, :, None]
    smp = np.uint8(smp_old * msk + (1 - msk) * smp_new)
    res[r:r + sz_smp, c:c + sz_smp] = smp

res = res[:sz_tar_f[0], :sz_tar_f[1]]

plt.imsave('./out/res1_1.jpg', res)
