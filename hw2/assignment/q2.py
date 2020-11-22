import cv2 as cv
import numpy as np
from scipy import signal
from sklearn import cluster

ship = cv.imread('./data/Greek_ship.jpg')
patch = cv.imread('./data/patch.png')


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


def find_candidates(src, tar, threshold_sigma, start, end, step):
    locs = list()
    ms = list()
    rs = list()
    for r in np.arange(start, end, step):
        r_patch = tar.copy()
        r_patch = cv.resize(r_patch, (0, 0), r_patch, r, r, interpolation=cv.INTER_AREA)

        m = match_template(src, r_patch)
        threshold = threshold_sigma * m.std() + m.mean()

        locs.append(np.argwhere(m > threshold))
        thresholded = m[m > threshold]
        ms.append(thresholded)
        rs.append(r * np.ones_like(thresholded))

    return np.concatenate(locs), np.concatenate(ms), np.concatenate(rs)


def find_matches(src, tar, confidence, down_sample, tar_start, tar_end, tar_step):
    src = src.copy()
    tar = tar.copy()
    src = cv.resize(src, (0, 0), src, down_sample, down_sample, interpolation=cv.INTER_AREA)
    tar = cv.resize(tar, (0, 0), src, down_sample, down_sample, interpolation=cv.INTER_AREA)

    cand_locations, cand_ratios, cand_sizes = find_candidates(src, tar, confidence, tar_start, tar_end, tar_step)

    bandwidth = cluster.estimate_bandwidth(cand_locations, quantile=0.3)
    algo = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    labels = algo.fit_predict(cand_locations)
    max_match_in_cluster = [np.max(cand_ratios[labels == i]) for i in range(labels.max() + 1)]

    res_loc = list()
    res_size = list()

    for loc, rate, sz, label in zip(cand_locations, cand_ratios, cand_sizes, labels):
        if rate == max_match_in_cluster[label]:
            res_loc.append((loc.astype(float) / down_sample).astype(int))
            res_size.append(sz)
    return res_loc, res_size


match_locations, sizes = find_matches(ship, patch, 2.9, 0.2, 0.5, 0.9, 0.1)

ship_cp = ship.copy()
cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0)]
for c, loc, sz in zip(range(len(sizes)), match_locations, sizes):
    cv.rectangle(ship_cp, (loc[1], loc[0]),
                 (loc[1] + int(patch.shape[1] * sz), loc[0] + int(patch.shape[0] * sz)), cols[c % len(cols)], 5)

cv.imwrite('./out/res03.jpg', ship_cp)
