# %%
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
import itertools
import random
from skimage.segmentation import slic, mark_boundaries

img = plt.imread('./data/slic.jpg')
r_img = cv.resize(img, (0, 0), img, 0.1, 0.1, cv.INTER_AREA)
plt.imshow(r_img)
plt.show()

# %%
src = cv.cvtColor(r_img, cv.COLOR_RGB2Lab).astype(float)
src = cv.GaussianBlur(src, (7, 7), 2)

K = 1024
m = 10
nb = 5
eps = 8
max_iters = 3
N = src.shape[0] * src.shape[1]
S = int(math.sqrt(N / K))

# generating gradient
sobel_x = cv.Sobel(src, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(src, cv.CV_64F, 0, 1, ksize=3)
grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
grad = np.mean(grad, axis=2)

# creating clusters
src = np.concatenate((np.dstack(np.mgrid[:src.shape[0], :src.shape[1]]), src), axis=2)
cluster_centers = np.array(list(itertools.product(np.arange(S, src.shape[0], S),
                                                  np.arange(S, src.shape[1], S))))

# perturbing
for clus in cluster_centers:
    mn = math.inf
    argmn = [0, 0]

    up = max(0, int(clus[0] - nb // 2))
    left = max(0, int(clus[1] - nb // 2))
    down = min(src.shape[0], int(clus[0] + nb // 2 + 1))
    right = min(src.shape[1], int(clus[1] + nb // 2 + 1))

    for i in range(up, down):
        for j in range(left, right):
            if grad[i, j] < mn:
                mn = grad[i, j]
                argmn = [i, j]
    clus[:2] = argmn

# generate color for cluster centers
cluster_centers = src[tuple(cluster_centers.T)]

# %%
plt.imshow(grad.astype(int), cmap='gray')
plt.show()
# %%
lbl = np.ones(src.shape[:2]) * -1
mtc = np.ones(src.shape[:2]) * math.inf
for it in range(max_iters):
    # lbl *= 0
    # lbl -= 1
    # mtc += math.inf
    # assign pixels to clusters
    for t, clus in enumerate(cluster_centers):
        up = max(0, int(clus[0] - S))
        left = max(0, int(clus[1] - S))
        down = min(src.shape[0], int(clus[0] + S + 1))
        right = min(src.shape[1], int(clus[1] + S + 1))
        for i in range(up, down):
            for j in range(left, right):
                px = src[i, j]
                dist_lab = np.linalg.norm(clus[2:] - px[2:])
                dist_xy = np.linalg.norm(clus[:2] - px[:2])
                dist = dist_lab + dist_xy * m / S
                if dist + eps < mtc[i, j]:
                    mtc[i, j] = dist
                    lbl[i, j] = t
    # recalculate cluster centers
    for t in range(len(cluster_centers)):
        lkup = src[lbl == t]
        if len(lkup) > 0:
            tmp = np.mean(lkup, axis=0)
            cluster_centers[t] = tmp
    print(f'end iteration {it}')

lbl_o = lbl.copy()
# %%
lbl = lbl_o.copy()
# lbl = morphology.opening(lbl, morphology.disk(3))
# lbl = morphology.closing(lbl, morphology.disk(3))

res = src.copy()

for t in range(len(cluster_centers)):
    res[lbl == t] = cluster_centers[t]

res = res[:, :, 2:].astype(np.uint8)
res = cv.cvtColor(res, cv.COLOR_Lab2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(r_img, lbl.astype(int)))
plt.show()
# %%
res2 = np.ones(res.shape[:2])
for t in range(len(cluster_centers)):
    res2[lbl == t] = random.randint(0, 255)
plt.figure(figsize=(10, 10))
plt.imshow(res2, cmap='gray')
plt.show()

# %%

segments_slic = slic(r_img, n_segments=1024, compactness=10, sigma=1)
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(r_img, segments_slic))
plt.show()
# %%
tsobel_x = cv.Sobel(lbl, cv.CV_64F, 1, 0, ksize=3)
tsobel_y = cv.Sobel(lbl, cv.CV_64F, 0, 1, ksize=3)
tgrad = np.sqrt(tsobel_x ** 2 + tsobel_y ** 2).astype(np.uint8)
cv.threshold(tgrad, 1, 255, cv.CV_8U, tgrad)
plt.imshow(tgrad, cmap='gray')
plt.show()
