# %%
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import feature, segmentation, util, exposure
from skimage import morphology as mph
import cv2 as cv

img = plt.imread('./data/birds.jpg')  # [1500:2500, 1500:4000]
# img = cv.resize(img,(0,0),img,0.2,0.2)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.scatter([1940],[250])
plt.show()

bird_pt = (1680, 3800)  # (180, 2300)  # 2
bird_pt2 = (1750, 3440)  # (250, 1940)  # 7

# %%

# img_eq = unsharp_mask(img,5,1)
img_eq = util.img_as_ubyte(exposure.equalize_adapthist(img, clip_limit=0.01))
plt.figure(figsize=(10, 10))
plt.imshow(img_eq)
plt.show()
# %%
segments_fz = segmentation.felzenszwalb(img, scale=400, sigma=1, min_size=150)
ccseg = len(np.unique(segments_fz))
print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
# %%
res = np.zeros_like(img)
for t in np.unique(segments_fz):
    res[segments_fz == t] = np.random.randint(0, 255, 3)
plt.figure(figsize=(10, 10))
# plt.imshow(mark_boundaries(img, segments_fz,mode='outer'))
plt.imshow(res)
plt.show()
# %%
ft_main = None
# HS histogram
img_hs = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)[..., :2]
hist_hs, _ = np.histogramdd(img_hs[segments_fz == segments_fz[bird_pt]], bins=(10, 5), range=((0, 256), (0, 100)),
                            density=True)
plt.imshow(np.log(hist_hs), interpolation='nearest', origin='low',
           extent=[_[1][0], _[1][-1], _[0][0], _[0][-1]])
# %%
lbp = feature.local_binary_pattern(rgb2gray(img), 16, 2, method='uniform').astype(int)
plt.imshow(lbp)
plt.show()
# %%
# hist_lbp,_ = np.histogram(lbp.reshape(-1,lbp.shape[-1]),bins=255,range=(0,256),density=True)
un, cn = np.unique(lbp[segments_fz == segments_fz[bird_pt]].ravel(), return_counts=True)
plt.scatter(un, cn)
# %%
ll = list()
for i in range(100):
    tt = np.argwhere(segments_fz == i)
    ll.append(np.abs(np.log((tt[..., 1].min() - tt[..., 1].max()) / (tt[..., 0].min() - tt[..., 0].max()))))
# ll = sorted(ll)
ll = np.array(ll)
print(ll)
plt.hist(ll, bins=20)
# %%
img_hs = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)[..., :2]

ftr_list = np.zeros((len(np.unique(segments_fz)), 50 + 18 + 1))
for i, row in enumerate(ftr_list):
    cond = segments_fz == i
    row[0:50] = np.histogramdd(img_hs[cond]
                               , bins=(10, 5), range=((0, 256), (0, 100))
                               , density=True)[0].ravel()
    un, cn = np.unique(lbp[cond].ravel(), return_counts=True)
    s = cn.sum()
    tt = np.argwhere(segments_fz == i)
    for u, c in zip(un, cn):
        row[u + 50] = c / s
    row[68] = (np.abs(np.log((tt[..., 1].min() - tt[..., 1].max()) / (tt[..., 0].min() - tt[..., 0].max()))))
print(ftr_list)


# %%
def dist(a, b):
    weights = (10, 1, 0.5)
    res = ((a - b) ** 2) / (a + b + 1e-12)
    res[0:50] *= weights[0]
    res[50:68] *= weights[1]
    res[68] *= weights[2]
    return res.sum()


# dist(ftr_list[0],ftr_list[1])
ll = list()
mtt = np.zeros((ftr_list.shape[0], ftr_list.shape[0]))
for i in range(ftr_list.shape[0]):
    for j in range(ftr_list.shape[0]):
        v = dist(ftr_list[i], ftr_list[j])
        if i < j:
            ll.append((v, i, j))
        mtt[i, j] = v
print(sorted(zip(mtt[2], range(68))))
print(mtt[2, 7])
plt.matshow(mtt)
# %%
res = np.zeros(img.shape[:2])
thr = 0.025
imgray = rgb2gray(img)
for i in range(ccseg):
    if mtt[2, i] < thr:
        res[segments_fz == i] = imgray[segments_fz == i]
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.show()
# %%

mask = np.where(res > 0, 1, 0).astype(np.uint8)
truc = mph.disk(6)
mask = mph.closing(mask, truc)
plt.figure(figsize=(10, 10))
plt.imshow(mask * imgray)


# %%
def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(32, 16), sharex='col', sharey='row')
    ax1.imshow(original, cmap='gray')
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap='gray')
    ax2.set_title(filter_name)
    ax2.axis('off')


mask = np.where(res > 0, 1, 0).astype(np.uint8)
plot_comparison(mph.closing(mask, mph.disk(15))[..., None] * img, mph.dilation(mask, truc)[..., None] * img, '')
