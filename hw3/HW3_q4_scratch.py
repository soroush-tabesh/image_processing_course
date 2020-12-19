# %%
from matplotlib import pyplot as plt
import numpy as np
from skimage import color as skcolor
from skimage import feature, segmentation, util, exposure, filters
from skimage import morphology as mph
import cv2 as cv


def imshow(src, **kwargs):
    plt.figure(dpi=200)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(src, **kwargs)


ratio = 0.3
img_o = plt.imread('./data/birds.jpg')  # [1500:2500, 1500:4000]
img_r = cv.resize(img_o, (0, 0), img_o, ratio, ratio)

imshow(img_r)

bird_pt = (np.array([[1670, 3800], [1750, 3440]]) * ratio).astype(int)  # (180, 2300)  # 2
plt.scatter(bird_pt[:, 1], bird_pt[:, 0], s=2, c='red')
plt.show()

# %%

img = img_r.copy()
# img = util.img_as_ubyte(exposure.equalize_adapthist(img,clip_limit=0.003))
# img = util.img_as_ubyte(filters.median(img,mph.cube(5)))
# img = util.img_as_ubyte(filters.unsharp_mask(img,5,1))
# img = util.img_as_ubyte(filters.gaussian(img,2))

imshow(img)
plt.show()

# %%

# segments_fz = segmentation.felzenszwalb(img, scale=400, sigma=0.5, min_size=150)
segments_fz = segmentation.felzenszwalb(img, scale=350, sigma=0.5, min_size=250)

c_seg = segments_fz.max()
print(f"Felzenszwalb number of segments: {c_seg}")

# %%

print(segments_fz[tuple(bird_pt[0])], segments_fz[tuple(bird_pt[1])])

# %%

label_image = np.zeros_like(img)
for t in np.unique(segments_fz):
    label_image[segments_fz == t] = np.random.randint(0, 255, 3)
plt.figure(figsize=(15, 15))
# imshow(mark_boundaries(img, segments_fz,mode='outer'))
imshow(label_image)
plt.scatter(bird_pt[:, 1], bird_pt[:, 0], s=2, c='red')
plt.show()

# %%

# HS histogram
img_hs = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)[..., :2]
print(tuple(bird_pt[0]), tuple(bird_pt[1]))
fig, ax = plt.subplots(ncols=2, sharey='col')
hist_hs, m_borders = np.histogramdd(img_hs[segments_fz == segments_fz[tuple(bird_pt[0])]], bins=(10, 5),
                                    range=((0, 256), (0, 100)), density=True)
ax[0].imshow(np.log(hist_hs + 1e-12), interpolation='nearest', origin='lower',
             extent=[m_borders[1][0], m_borders[1][-1], m_borders[0][0], m_borders[0][-1]])
hist_hs, m_borders = np.histogramdd(img_hs[segments_fz == segments_fz[tuple(bird_pt[1])]], bins=(10, 5),
                                    range=((0, 256), (0, 100)), density=True)
ax[1].imshow(np.log(hist_hs + 1e-12), interpolation='nearest', origin='lower',
             extent=[m_borders[1][0], m_borders[1][-1], m_borders[0][0], m_borders[0][-1]])
plt.show()

# %%

lbp = feature.local_binary_pattern(skcolor.rgb2gray(img), 8, 1, method='uniform').astype(int)
imshow(lbp, cmap='Set2')
plt.show()

# %%

# hist_lbp,_ = np.histogram(lbp.reshape(-1,lbp.shape[-1]),bins=255,range=(0,256),density=True)
un, cn = np.unique(lbp[segments_fz == segments_fz[tuple(bird_pt[0])]].ravel(), return_counts=True)
plt.scatter(un, cn)
un, cn = np.unique(lbp[segments_fz == segments_fz[tuple(bird_pt[1])]].ravel(), return_counts=True)
plt.scatter(un, cn)
un, cn = np.unique(lbp[segments_fz == 2].ravel(), return_counts=True)
plt.scatter(un, cn)
plt.show()
print(len(un))


# %%


def get_aspect(label):
    tt = np.argwhere(segments_fz == label)
    return np.abs(np.log((tt[..., 1].min() - tt[..., 1].max()) / (tt[..., 0].min() - tt[..., 0].max())))


ll = list()
for i in range(100):
    ll.append(get_aspect(i))
ll = np.array(ll)
print(get_aspect(5), get_aspect(8))
plt.hist(ll, bins=15)
plt.show()


# %%

def get_box_around(label):
    cond = segments_fz == label
    t = np.argwhere(cond)
    u = t[..., 0].min()
    d = t[..., 0].max()
    l = t[..., 1].min()
    r = t[..., 1].max()
    return img[u:d + 1, l:r + 1] * cond[u:d + 1, l:r + 1]


img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)


def get_features(label):
    ft = list()
    cond = segments_fz == label

    ft.append(np.histogramdd(img_hsv[cond][..., :2]
                             , bins=(10, 5), range=((0, 256), (0, 100))
                             , density=True)[0].ravel())

    un, cn = np.unique(lbp[cond].ravel(), return_counts=True)
    tmp = np.zeros((18))
    s = cn.sum()
    for u, c in zip(un, cn):
        tmp[u] = c / s
    ft.append(tmp)

    ft.append(get_aspect(label))

    ft.append(img_hsv[cond][2].var())
    return ft


ftr_list = [get_features(i) for i in range(c_seg)]


# %%

def distance(fa, fb):
    weights = (10, 1.5, 0.5, 0.0002)
    res = 0
    for i, a, b in zip(range(len(fa)), fa, fb):
        res += np.sum(((a - b) ** 2) / (a + b + 1e-12) * weights[i])
    return res


mtt = np.zeros((c_seg, c_seg))
for i, e in np.ndenumerate(mtt):
    mtt[i] = distance(ftr_list[i[0]], ftr_list[i[1]])

plt.matshow(mtt)
plt.show()

# %%

label_image = np.zeros(img.shape[:2])
thr = 0.22
img_gray = skcolor.rgb2gray(img)
for i in range(c_seg):
    if mtt[2, i] < thr:
        cond = segments_fz == i
        cond = mph.opening(cond, selem=mph.disk(3))
        cond = mph.dilation(cond, selem=mph.disk(2))
        label_image[cond] = i
# img_overlay = img.copy()
# img_overlay[..., 1] *= label_image == 0

img_overlay = skcolor.label2rgb(label=label_image, image=img, bg_label=0)

# imshow(label_image)
# plt.show()
imshow(img_overlay)
plt.show()
