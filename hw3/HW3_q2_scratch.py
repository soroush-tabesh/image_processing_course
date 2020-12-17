# %%
from sklearn import cluster
from sklearn import preprocessing as prp
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

img = plt.imread('./data/park.jpg')
r_img = cv.resize(img, (0, 0), img, 0.1, 0.1, cv.INTER_AREA)
plt.imshow(r_img)
plt.show()


# %%
def segment_cluster(src, ratio, bandwidth=None, r_src=None):
    f_src = np.reshape(src, (src.shape[0] * src.shape[1], src.shape[2]))
    cf_src = np.concatenate((f_src, np.arange(f_src.shape[0]).reshape((f_src.shape[0], 1))), axis=1)
    if r_src is None:
        r_src = cv.resize(src, (0, 0), src, ratio, ratio, cv.INTER_AREA)
    cr_src = np.concatenate((r_src, np.arange(r_src.shape[0] * r_src.shape[1])
                             .reshape((r_src.shape[0], r_src.shape[1], 1))), axis=2)
    fcr_src = np.reshape(cr_src, (cr_src.shape[0] * cr_src.shape[1], cr_src.shape[2]))
    trans = prp.StandardScaler()
    tfcr_src = trans.fit_transform(fcr_src)
    if bandwidth is None:
        bandwidth = cluster.estimate_bandwidth(tfcr_src, quantile=0.3, n_samples=3000, n_jobs=-1)
        print(bandwidth)
    ms = cluster.MeanShift(bandwidth=bandwidth, n_jobs=-1, max_iter=100, bin_seeding=True)
    ms = ms.fit(tfcr_src)
    res = ms.cluster_centers_[ms.predict(trans.transform(cf_src))][:, :, :2]
    return np.reshape(res, src.shape).astype(np.uint8)


# %%

rs = segment_cluster(img, 1, r_src=r_img)
plt.imshow(rs)
plt.show()

# %%
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i, bd in enumerate([20, 30, 35, 40, 50, 60, 80, 100, 120]):
    ax[i // 3][i % 3].imshow(segment_cluster(img, 1, r_src=r_img))
plt.show()

# %%

src = img
r_src = r_img

c_src = np.concatenate((src, np.dstack(np.mgrid[:src.shape[0], :src.shape[1]]) // 5), axis=2)
fc_src = np.reshape(c_src, (src.shape[0] * src.shape[1], src.shape[2] + 2))

cr_src = np.concatenate((r_src, np.dstack(np.mgrid[:r_src.shape[0], :r_src.shape[1]]) * 2), axis=2)
fcr_src = np.reshape(cr_src, (r_src.shape[0] * r_src.shape[1], r_src.shape[2] + 2))

# trans = prp.RobustScaler()
# tfcr = trans.fit_transform(fcr_src)

bandwidth = cluster.estimate_bandwidth(fcr_src, quantile=0.3, n_samples=3000, n_jobs=-1)
print(bandwidth)

ms = cluster.MeanShift(bandwidth=bandwidth / 4, n_jobs=-1, max_iter=150, bin_seeding=True)
ms = ms.fit(fcr_src)

res = ms.cluster_centers_[ms.predict(fc_src)][:, :3]
res = np.reshape(res, src.shape).astype(np.uint8)

plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.show()

# %%

dist_coef = 0.1
bandwidth_coef = 0.23
with_xy = False

feat_src = cv.cvtColor(img, cv.COLOR_RGB2BGR).astype(float)
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

plt.scatter(r_feat[:, 0], r_feat[:, 1], s=0.01, c=ms.labels_)
plt.show()
plt.scatter(r_feat[:, 0], r_feat[:, 2], s=0.01, c=ms.labels_)
plt.show()
plt.scatter(r_feat[:, 1], r_feat[:, 2], s=0.01, c=ms.labels_)
plt.show()

res = ms.cluster_centers_[ms.predict(feat_src)][:, :3]
res = np.reshape(res, img.shape).astype(np.uint8)

plt.figure(figsize=(10, 10))
plt.title(f'with_xy={with_xy} , '
          f'dist_coef={dist_coef} , '
          f'bandwidth_coef={bandwidth_coef} , '
          f'est_bandwidth={bandwidth:.2f} , ')
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
plt.show()
