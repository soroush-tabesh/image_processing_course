# %% imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from scipy import signal

# %% load image
pic_orig = cv.imread('./data/melons.tif', cv.IMREAD_ANYDEPTH)

# %% resize image for debug purpose
res_ratio = 1
pic = cv.resize(pic_orig, (0, 0), pic_orig, res_ratio, res_ratio, interpolation=cv.INTER_AREA)

# %% show original image
plt.figure(figsize=(10, 30))
plt.imshow(pic, cmap='gray')
plt.show()


# %% the registration algorithm
def find_translation(src, tar, ratio, step_offset, min_size) -> (np.ndarray, float):
    """
    src and tar should be the same in size
    :param src: source image to find 'tar' on
    :param tar: target image to find in 'src'
    :param ratio: resize ratio of gaussian pyramid
    :param step_offset: search size on each step
    :param min_size: threshold size of image to perform brute-force search
    :return: A tuple showing how much translation should be applied to 'tar' to get 'src' plus a float showing match
        percentage
    """
    if max(src.shape + tar.shape) < min_size:
        # perform a brute-force search
        print('Brute-force search start at top of the pyramid')
        src = (src - np.mean(src)) / np.std(src)
        tar = (tar - np.mean(tar)) / np.std(tar)
        corr = signal.correlate2d(src, tar)
        offsets = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        print(f'Search end at size {src.shape}')
        return offsets - np.array(tar.shape), np.amax(corr) / src.size
    else:
        # recursively solve then adjust
        kernel_size = 2 * round(1 / ratio) + 1
        sigma = 1 / ratio
        offset, _ = find_translation(
            cv.GaussianBlur(cv.resize(src, (0, 0), src, ratio, ratio, interpolation=cv.INTER_AREA),
                            (kernel_size, kernel_size), sigma),
            cv.GaussianBlur(cv.resize(tar, (0, 0), tar, ratio, ratio, interpolation=cv.INTER_AREA),
                            (kernel_size, kernel_size), sigma),
            ratio,
            step_offset,
            min_size)
        print(f'Fine tuning at size {src.shape}')
        offset *= 2
        mx = 0
        ind = (0, 0)
        src = (src - np.mean(src)) / np.std(src)
        tar = (tar - np.mean(tar)) / np.std(tar)
        for i_off in range(-step_offset, step_offset + 1):
            for j_off in range(-step_offset, step_offset + 1):
                t_tar = np.roll(tar, offset + (i_off, j_off), (0, 1))
                corr = abs((t_tar * src).sum())
                if corr > mx:
                    mx = corr
                    ind = (i_off, j_off)
        return offset + ind, mx


# %% image partitioning
t_wd = pic.shape[0] // 3
pic_b = pic[0:t_wd].astype(float)
pic_g = pic[t_wd:2 * t_wd].astype(float)
pic_r = pic[2 * t_wd:t_wd * 3].astype(float)

# %% image registration process
m_ratio = 0.5
m_step_offset = 2
m_min_size = 200

print(f'----start registration of green channel {time.process_time()}')
g_offset, _1 = find_translation(pic_r.copy(), pic_g.copy(), m_ratio, m_step_offset, m_min_size)
print(f'start registration of blue channel {time.process_time()}')
b_offset, _2 = find_translation(pic_r.copy(), pic_b.copy(), m_ratio, m_step_offset, m_min_size)

pic_g = np.roll(pic_g, g_offset, (0, 1))
pic_b = np.roll(pic_b, b_offset, (0, 1))
pic_res = np.stack((pic_r / 256, pic_g / 256, pic_b / 256), axis=2)

# %% show result

plt.imshow(pic_res.astype(int))
plt.show()
