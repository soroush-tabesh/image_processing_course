from sklearn import cluster
import numpy as np
import cv2 as cv

img = cv.imread('./data/park.jpg')
r_img = cv.resize(img, (0, 0), img, 0.1, 0.1, cv.INTER_AREA)

dist_coef = 0.1
bandwidth_coef = 0.23
with_xy = False

feat_src = img.astype(float)
feat_src = cv.GaussianBlur(feat_src, (71, 71), 0)

if with_xy:
    coord_table = np.dstack(np.mgrid[:feat_src.shape[0], :feat_src.shape[1]]) * dist_coef
    feat_src = np.concatenate((feat_src, coord_table), axis=2)

r_feat = feat_src[::50, ::50]
# r_feat = cv.resize(feat_src, (0, 0), feat_src, 0.2, 0.2, cv.INTER_AREA)

feat_src = feat_src.reshape((feat_src.shape[0] * feat_src.shape[1], feat_src.shape[2]))
r_feat = r_feat.reshape((r_feat.shape[0] * r_feat.shape[1], r_feat.shape[2]))

bandwidth = cluster.estimate_bandwidth(feat_src, quantile=0.3, n_samples=3000, n_jobs=-1)

ms = cluster.MeanShift(bandwidth=bandwidth * bandwidth_coef, n_jobs=-1, max_iter=100, bin_seeding=True)
ms = ms.fit(r_feat)

res = ms.cluster_centers_[ms.predict(feat_src)][:, :3]
res = np.reshape(res, img.shape).astype(np.uint8)

cv.imwrite('./out/res04.jpg', res)
