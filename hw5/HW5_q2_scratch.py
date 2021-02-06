# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import util, color


def imshow(*imgs):
    plt.close()
    fig, axs = plt.subplots(ncols=len(imgs))
    if len(imgs) == 1:
        plt.imshow(imgs[0])
    else:
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
    plt.show()


face2 = util.img_as_float64(plt.imread('./data/images/biden.jpg'))
face1 = util.img_as_float64(plt.imread('./data/images/trump.jpg'))
face_mask = color.rgb2gray(plt.imread('./data/images/mask_trump_biden.png')[:, :, :3])

imshow(face1, face2, face_mask)

# %%

def pyr_up(src, cutoff, ratio):
    lowpassed = cv.GaussianBlur(src, (2 * cutoff + 1, 2 * cutoff + 1), 0,
                                borderType=cv.BORDER_REFLECT101)
    src -= lowpassed
    return cv.resize(lowpassed, (0, 0), None, ratio, ratio, cv.INTER_AREA)


def blend(src, tar, mask, bandwidth):
    mask = cv.GaussianBlur(mask, (2 * bandwidth + 1, 2 * bandwidth + 1), 0
                           , borderType=cv.BORDER_REFLECT101)[:, :, None]
    return src * mask + tar * (1 - mask)


iterations = 8
m_ratio = 0.8
bandwidth_low = 20
bandwidth_high = 40
m_cutoff = 8

pyr_lap_1 = [face1.copy()]
pyr_lap_2 = [face2.copy()]
pyr_mask = [face_mask.copy()]

for i in range(iterations):
    pyr_lap_1.append(pyr_up(pyr_lap_1[i], m_cutoff, m_ratio))
    pyr_lap_2.append(pyr_up(pyr_lap_2[i], m_cutoff, m_ratio))
    pyr_mask.append(cv.resize(pyr_mask[i], (0, 0), None, m_ratio, m_ratio, cv.INTER_NEAREST))

pyr_lap_1[iterations] = blend(pyr_lap_1[iterations], pyr_lap_2[iterations], pyr_mask[iterations], bandwidth_low)

for i in range(iterations - 1, -1, -1):
    pyr_lap_1[i] = blend(pyr_lap_1[i], pyr_lap_2[i], pyr_mask[i], bandwidth_high)
    tmp = cv.resize(pyr_lap_1[i + 1], pyr_lap_1[i].shape[:2][::-1], interpolation=cv.INTER_AREA)
    pyr_lap_1[i] += tmp

res = np.clip(pyr_lap_1[0], 0, 1)
imshow(res)
plt.imsave('./tt2.png', res)
# %%
msk = face_mask.copy()
msk = cv.GaussianBlur(msk, (201, 201), 0
                      , borderType=cv.BORDER_REFLECT101)[:, :, None]
ress = face1 * msk + face2 * (1 - msk)
imshow(ress)
