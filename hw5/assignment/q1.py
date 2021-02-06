import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import util, color
from scipy import sparse
from scipy.sparse.linalg import spsolve

src = util.img_as_float64(plt.imread('1.source.jpg'))
tar = util.img_as_float64(plt.imread('2.target.jpg'))
mask = color.rgb2gray(color.rgba2rgb(plt.imread('mask_trump_biden.png')))[:, :, None]

mask_arg = np.argwhere(mask > 0.01)
c_area = (mask_arg[:, 0].min() - 2, mask_arg[:, 1].min() - 2, mask_arg[:, 0].max() + 3, mask_arg[:, 1].max() + 3)

src_r = src[c_area[0]:c_area[2], c_area[1]:c_area[3]].copy()
tar_r = tar[c_area[0]:c_area[2], c_area[1]:c_area[3]].copy()
mask_r = mask[c_area[0]:c_area[2], c_area[1]:c_area[3]].copy()

tar_r_lap = cv.Laplacian(tar_r, cv.CV_64F)

shape_r = tar_r.shape
size = shape_r[0] * shape_r[1]

diags = np.zeros((5, size))
offsets = [0, -1, 1, -shape_r[1], shape_r[1]]

ind_mask = np.argwhere(mask_r > 0.01)
ind_mask = ind_mask[:, 0] * shape_r[1] + ind_mask[:, 1]

diags[0, :] = 1
diags[:, ind_mask] = 1
diags[0, ind_mask] = -4
for i, offset in enumerate(offsets):
    diags[i, :] = np.roll(diags[i, :], offset)

mat = sparse.dia_matrix((diags, offsets), (size, size))
vec = src_r.copy().reshape((size, 3))
vec[ind_mask] = tar_r_lap.reshape((size, 3))[ind_mask]

res_r = spsolve(mat.tocsr(), vec)
res_r = res_r.reshape(shape_r)

res = src.copy()
res[c_area[0]:c_area[2], c_area[1]:c_area[3]] = res_r

res = np.clip(res, 0, 1)

plt.imsave('res1.jpg', res)
