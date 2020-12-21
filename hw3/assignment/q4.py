from matplotlib import pyplot as plt
import numpy as np
from skimage import color as skcolor
from skimage import feature, segmentation
from skimage import morphology as mph
import cv2 as cv

ratio = 0.3
img_o = plt.imread('./data/birds.jpg')
img_r = cv.resize(img_o, (0, 0), img_o, ratio, ratio)

bird_pt = (np.array([[1670, 3800]]) * ratio).astype(int)  # hardcoded bird

img = img_r.copy()

segments_fz = segmentation.felzenszwalb(img, scale=350, sigma=0.5, min_size=250)
c_seg = segments_fz.max()
img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)
lbp = feature.local_binary_pattern(skcolor.rgb2gray(img), 8, 1, method='uniform').astype(int)


def get_aspect(label):
    tt = np.argwhere(segments_fz == label)
    return np.abs(np.log((tt[..., 1].min() - tt[..., 1].max()) / (tt[..., 0].min() - tt[..., 0].max())))


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


def distance(fa, fb):
    weights = (10, 1.5, 0.5, 0.0002)
    res = 0
    for i, a, b in zip(range(len(fa)), fa, fb):
        res += np.sum(((a - b) ** 2) / (a + b + 1e-12) * weights[i])
    return res


ftr_list = [get_features(i) for i in range(c_seg)]
mtt = np.zeros((c_seg, c_seg))
for i, e in np.ndenumerate(mtt):
    mtt[i] = distance(ftr_list[i[0]], ftr_list[i[1]])

label_image = np.zeros(img.shape[:2])
thr = 0.22
for i in range(c_seg):
    if mtt[2, i] < thr:
        cond = segments_fz == i
        cond = mph.opening(cond, selem=mph.disk(3))
        cond = mph.dilation(cond, selem=mph.disk(2))
        label_image[cond] = i

img_overlay = skcolor.label2rgb(label=label_image, image=img, bg_label=0)

plt.imsave('./out/res09.jpg', img_overlay)
plt.show()
