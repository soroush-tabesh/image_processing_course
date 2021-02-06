# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import util, color
from scipy import sparse
from scipy.sparse.linalg import spsolve


def imshow(*imgs):
    plt.close()
    fig, axs = plt.subplots(ncols=len(imgs))
    if len(imgs) == 1:
        plt.imshow(imgs[0])
    else:
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
    plt.show()


pic_trump = util.img_as_float64(plt.imread('./data/images/biden.jpg'))
pic_biden = util.img_as_float64(plt.imread('./data/images/trump.jpg'))
mask = color.rgb2gray(color.rgba2rgb(plt.imread('./data/images/mask_trump_biden.png')))[:, :, None]

imshow(pic_biden, pic_trump, pic_biden * mask + pic_trump * (1 - mask))

# %%
c_area = np.argwhere(mask > 0.001)
c_area = (c_area[:, 0].min() - 2, c_area[:, 1].min() - 2, c_area[:, 0].max() + 3, c_area[:, 1].max() + 3)

pic_trump_r = pic_trump[c_area[0]:c_area[2], c_area[1]:c_area[3]].copy()
pic_biden_r = pic_biden[c_area[0]:c_area[2], c_area[1]:c_area[3]].copy()
mask_r = mask[c_area[0]:c_area[2], c_area[1]:c_area[3]].copy()
imshow(pic_biden_r, pic_trump_r, pic_biden_r * mask_r + pic_trump_r * (1 - mask_r))

pic_biden_r_lap = cv.Laplacian(pic_biden_r, cv.CV_64F)
pic_trump_r_lap = cv.Laplacian(pic_trump_r, cv.CV_64F)

tmsk = cv.GaussianBlur(mask_r, (21, 21), 0)[...,None]
pic_biden_r_lap = tmsk * pic_biden_r_lap + (1 - tmsk) * pic_trump_r_lap

# %%
shape_r = pic_biden_r.shape
size = shape_r[0] * shape_r[1]

diags = np.zeros((5, size))
offsets = [0, -1, 1, -shape_r[1], shape_r[1]]

ind_os = np.argwhere(mask_r > 0.01)
ind_os = ind_os[:, 0] * shape_r[1] + ind_os[:, 1]

diags[0, :] = 1
diags[:, ind_os] = 1
diags[0, ind_os] = -4
for i, offset in enumerate(offsets):
    diags[i, :] = np.roll(diags[i, :], offset)

mat = sparse.dia_matrix((diags, offsets), (size, size))
vec = pic_trump_r.copy().reshape((size, 3))
vec[ind_os] = pic_biden_r_lap.reshape((size, 3))[ind_os]

res_r = spsolve(mat.tocsr(), vec)
res_r = res_r.reshape(shape_r)

res = pic_trump.copy()
res[c_area[0]:c_area[2], c_area[1]:c_area[3]] = res_r

res = np.clip(res, 0, 1)
imshow(res)
# %%
plt.imsave('./tt1.png', res)
