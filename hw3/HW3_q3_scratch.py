# %%
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
import itertools
import random

img = plt.imread('./data/slic.jpg')
r_img = cv.resize(img, (0, 0), img, 0.1, 0.1, cv.INTER_AREA)
plt.imshow(r_img)
plt.show()

# %%
src = cv.cvtColor(r_img, cv.COLOR_RGB2Lab).astype(float)
K = 500
m = 15
nb = 5
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
    for i in range(clus[0] - nb // 2, clus[0] + nb // 2 + 1):
        for j in range(clus[1] - nb // 2, clus[1] + nb // 2 + 1):
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
for it in range(7):
    lbl *= 0
    lbl -= 1
    mtc += math.inf
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
                if dist < mtc[i, j]:
                    mtc[i, j] = dist
                    lbl[i, j] = t
    # recalculate cluster centers
    for t in range(len(cluster_centers)):
        cluster_centers[t] = np.mean(src[lbl == t], axis=0)
    print(f'end iteration {it}')
# %%
res = src.copy()

for t in range(len(cluster_centers)):
    res[lbl == t] = cluster_centers[t]

res = res[:, :, 2:].astype(np.uint8)
res = cv.cvtColor(res, cv.COLOR_Lab2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.show()
# %%
res2 = np.ones(res.shape[:2])
for t in range(len(cluster_centers)):
    res2[lbl == t] = random.randint(0, 255)
plt.figure(figsize=(10, 10))
plt.imshow(res2, cmap='gray')
plt.show()
#%%
