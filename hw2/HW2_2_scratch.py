# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import cluster
import math

ship = cv.cvtColor(cv.imread('./data/Greek_ship.jpg'), cv.COLOR_BGR2RGB)
patch = cv.cvtColor(cv.imread('./data/patch.png'), cv.COLOR_BGR2RGB)

res_ratio = 0.3
ship = cv.resize(ship, (0, 0), ship, res_ratio, res_ratio, interpolation=cv.INTER_AREA)
patch = cv.resize(patch, (0, 0), patch, res_ratio, res_ratio, interpolation=cv.INTER_AREA)


# %%
def find_match(src, tar):
    mx = (0, 0, 0)
    for r in np.linspace(0.4, 1.4, 10):
        r_patch = tar.copy()
        r_patch = cv.resize(r_patch, (0, 0), r_patch, r, r, interpolation=cv.INTER_AREA)
        m = cv.matchTemplate(src, r_patch, cv.TM_CCOEFF_NORMED)
        # m = match_template(src, r_patch)
        rate = m.max()
        if rate > mx[2]:
            mx = (np.unravel_index(np.argmax(m), m.shape), r, rate)
    return mx


def find_all_match(src, tar, threshold):
    match_list = list()
    src = src.copy()
    tar = tar.copy()
    while True:
        arg, resize, rate = find_match(src, tar)
        if rate < threshold:
            break
        print(rate)
        match_list.append((arg, resize))
        cv.rectangle(src, (arg[1], arg[0]),
                     (arg[1] + int(tar.shape[1] * resize), arg[0] + int(tar.shape[0] * resize)), (0, 0, 0), -1)
    return match_list


res = find_all_match(ship, patch, 0.65)

img = ship.copy()
for elem in res:
    cv.rectangle(img, (elem[0][1], elem[0][0]),
                 (elem[0][1] + int(patch.shape[1] * elem[1]), elem[0][0] + int(patch.shape[0] * elem[1]))
                 , (0, 0, 255), 5)

plt.imshow(img)
plt.show()


# %% using clustering

def match_template(src, tar):
    src = src.astype(np.float)
    tar = tar.astype(np.float)
    tar -= tar.mean()

    tar2 = tar ** 2
    src_avg = signal.correlate(src, np.ones_like(tar) / float(tar.size), mode='valid', method='fft')
    src2_integral = signal.correlate(src ** 2, np.ones_like(tar), mode='valid', method='fft')
    src_avg2 = src_avg ** 2
    cc = signal.correlate(src, tar, mode='valid', method='fft')

    ncc = cc / np.sqrt(tar2.sum() * (src2_integral - tar.size * src_avg2))

    return ncc


def match_template_multi(src, tar):
    m0 = match_template(src[:, :, 0], tar[:, :, 0])
    m1 = match_template(src[:, :, 1], tar[:, :, 1])
    m2 = match_template(src[:, :, 2], tar[:, :, 2])
    return (m0 + m1 + m2) / 3
    # return np.power(m0 * m1 * m2, 1 / 3)


def find_candidates(src, tar, threshold, threshold_sigma):
    locs = list()
    ms = list()
    rs = list()
    for r in [0.5, 0.4, 0.7]:
        r_patch = tar.copy()
        # r_patch = cv.GaussianBlur(r_patch, (3, 3), 0)
        r_patch = cv.resize(r_patch, (0, 0), r_patch, r, r, interpolation=cv.INTER_AREA)

        # m = match_template(src, r_patch)
        m = match_template(src, r_patch)
        # m = cv.matchTemplate(src, r_patch, cv.TM_CCOEFF_NORMED)
        # threshold = threshold_sigma * m.std() + m.mean()

        locs.append(np.argwhere(m > threshold))
        thresholded = m[m > threshold]
        ms.append(thresholded)
        rs.append(r * np.ones_like(thresholded))

    return np.concatenate(locs), np.concatenate(ms), np.concatenate(rs)


def classify_matches(cands):
    pass


ship_cp = ship.copy()
msk = np.zeros_like(ship)
locs, ms, rs = find_candidates(ship, patch, 0.65, 2)
for loc, pers, sz in zip(locs, ms, rs):
    cv.rectangle(ship_cp, (loc[1], loc[0]),
                 (loc[1] + int(patch.shape[1] * sz), loc[0] + int(patch.shape[0] * sz)), (0, 0, 255), 2)
    msk[loc[0], loc[1]] = (255, 0, 0)
plt.imshow(ship_cp)
plt.show()
plt.imshow(msk)
plt.show()
# %% clustering

# todo blur the image

ship_cp = ship.copy()
locs, ms, rs = find_candidates(ship, patch, 0.65, 2)
msk = np.zeros_like(ship)

bandwidth = cluster.estimate_bandwidth(locs, quantile=0.3)
algo = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
lbls = algo.fit_predict(locs)
cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0)]
mxx = [np.max(ms[lbls == i]) for i in range(lbls.max() + 1)]
for loc, pers, sz, c in zip(locs, ms, rs, lbls):
    msk[loc[0], loc[1]] = (255, 0, 0)
    print(c)
    if pers != mxx[c]:
        continue
    cv.rectangle(ship_cp, (loc[1], loc[0]),
                 (loc[1] + int(patch.shape[1] * sz), loc[0] + int(patch.shape[0] * sz)), cols[c % 7], 2)
plt.imshow(ship_cp)
plt.show()
plt.imshow(msk)
plt.show()
# %%
r_patch = patch.copy()
r = 0.5
r_patch = cv.resize(r_patch, (0, 0), r_patch, r, r, interpolation=cv.INTER_AREA)
# mrs = match_template_multi(ship, r_patch)
mrs = match_template(ship, r_patch)
plt.imshow(mrs, cmap='viridis')
plt.show()
# %%
mrs_blt = cv.matchTemplate(ship, r_patch, cv.TM_CCOEFF_NORMED)
plt.imshow(mrs_blt, cmap='viridis')
plt.show()
