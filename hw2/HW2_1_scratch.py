# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

img_orig = cv.cvtColor(cv.imread('./data/flowers_blur.png'), cv.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_orig)
plt.show()


# %%
def laplacian_of_gaussian(x, y, sigma: float):
    return (x ** 2 + y ** 2 - 2 * sigma ** 2) \
           * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) \
           / (2 * math.pi * sigma ** 6)


def unsharp_filter(ksize: int, sigma: float):
    res = np.zeros((ksize, ksize))
    for i in range(-ksize // 2, ksize // 2 + 1):
        for j in range(-ksize // 2, ksize // 2 + 1):
            res[i + ksize // 2][j + ksize // 2] = laplacian_of_gaussian(j, i, sigma)
    return res - res.mean()


def sharpen(src, coef, ksize, sigma):
    flt = unsharp_filter(ksize, sigma)
    src = src.astype(np.float)
    lap = cv.filter2D(src, -1, flt).astype(float)
    res = np.clip(src + coef * lap, 0, 255).astype(np.uint8)
    return res, lap


def builtin_sharpen(src, coef, ksize, sigma):
    gus = cv.GaussianBlur(src, (ksize, ksize), sigma)
    lap = cv.Laplacian(gus, ddepth=cv.CV_32F, ksize=ksize)
    lap /= np.max(lap)
    return np.clip(src + coef * lap, 0, 255).astype(np.uint8)


# %%
def normalize_image(src):
    src = src.copy().astype(float)
    src -= src.mean()
    src /= src.std()
    src *= 50
    src += 127
    return np.clip(src, 0, 255).astype(np.uint8)


m_coef = -20
m_ksize = 5
m_sigma = 0.73

img = img_orig.copy()
img, llp = sharpen(img, m_coef, m_ksize, m_sigma)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()
plt.figure(figsize=(10, 10))
plt.imshow(normalize_image(llp))
plt.show()
# %%

cmpr = 2
cv.imwrite(f'blabla-{m_coef}-{m_ksize}-{cmpr}.png', cv.cvtColor(img, cv.COLOR_RGB2BGR)
           , [int(cv.IMWRITE_PNG_COMPRESSION), cmpr])
