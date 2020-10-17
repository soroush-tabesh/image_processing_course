# %% imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# %% load image
res_ratio = 0.25
pic_orig = cv.imread('./data/melons.tif')
pic_orig = cv.resize(pic_orig, (0, 0), pic_orig, res_ratio, res_ratio, interpolation=cv.INTER_AREA)

plt.gray()
plt.figure(dpi=200)
plt.imshow(pic_orig)
plt.show()

