import cv2 as cv
import numpy as np
import math

img_orig = cv.imread('./data/flowers_blur.png')


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


m_coef = -20
m_ksize = 5
m_sigma = 0.73

img, lapped = sharpen(img_orig, m_coef, m_ksize, m_sigma)


def normalize_image(src):
    src = src.copy().astype(float)
    src -= src.mean()
    src /= src.std()
    src *= 50
    src += 127
    return np.clip(src, 0, 255).astype(np.uint8)


cv.imwrite('./out/res01.jpg', normalize_image(lapped))
cv.imwrite('./out/res02.jpg', img)
