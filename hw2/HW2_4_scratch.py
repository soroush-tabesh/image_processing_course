# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

near = cv.cvtColor(cv.imread('./data/q4_01_near.jpg'), cv.COLOR_BGR2RGB)
far = cv.cvtColor(cv.imread('./data/q4_02_far.jpg'), cv.COLOR_BGR2RGB)

plt.imshow(near)
plt.show()
plt.imshow(far)
plt.show()
# %% registering pics

# x,y coords
# near
cue_points_near = np.array([
    (179, 160),  # right eye
    (108, 157),  # left eye
    (131, 223),  # nose bottom
    # (134, 250)  # mid lip
])
# far
cue_points_far = np.array([
    (176, 161),
    (108, 160),
    (130, 225),
    # (133, 250)
])

t_mat, _ = cv.estimateAffine2D(cue_points_near, cue_points_far)
near = cv.warpAffine(near, t_mat, far.shape[:2][::-1], flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
plt.imshow(near)
plt.show()


# save 03,04

# %%
def calc_log_normalized_mag(src):
    src = np.log(np.abs(src))
    src += src.min()
    src /= src.max()
    return src


def calc_dft(src):
    dft_full = list()
    for i in range(3):
        dft = np.fft.fft2(src[:, :, i])
        dft = np.fft.fftshift(dft)
        dft_full.append(dft)
    return np.stack(dft_full, axis=2)


near_dft = calc_dft(near)
far_dft = calc_dft(far)

# save 05,06
plt.imshow(calc_log_normalized_mag(near_dft))
plt.show()
plt.imshow(calc_log_normalized_mag(far_dft))
plt.show()


# %% make filters

def binormal_pdf(pos, sigma, mean):
    pos = np.array(pos) - np.array(mean)
    return math.exp(-((pos * pos).sum()) / sigma ** 2 / 2) / (2 * math.pi * sigma ** 2)


def gaussian_filter_shfited(dims, sigma):
    res = np.zeros(dims)
    mean = np.array(dims) / 2
    for loc in np.ndindex(dims):
        res[loc] = binormal_pdf(loc, sigma, mean)
    return res / res.max()


lowpass_sigma = 12
highpass_sigma = 34

lowpass_filter = gaussian_filter_shfited(far.shape[:2], lowpass_sigma)
highpass_filter = 1 - gaussian_filter_shfited(near.shape[:2], highpass_sigma)

# save 07,08
plt.imshow(lowpass_filter, cmap='gray')
plt.show()
plt.imshow(highpass_filter, cmap='gray')
plt.show()


# %% cutoff filters

def cutoff_filter(dims, threshold):
    res = np.zeros(dims)
    mean = np.array(dims) / 2
    for loc in np.ndindex(dims):
        dist = np.array(loc) - mean
        dist = np.linalg.norm(dist)
        if dist < threshold:
            res[loc] = 1
    return res


lowpass_cutoff = 22
highpass_cutoff = 18

lowpass_filter_cut = cutoff_filter(lowpass_filter.shape, lowpass_cutoff) * lowpass_filter
highpass_filter_cut = (1 - cutoff_filter(highpass_filter.shape, highpass_cutoff)) * highpass_filter

# save 09,10
plt.imshow(lowpass_filter_cut, cmap='gray')
plt.show()
plt.imshow(highpass_filter_cut, cmap='gray')
plt.show()

# %% apply filters

lowpassed = far_dft * np.repeat(lowpass_filter_cut[:, :, np.newaxis], 3, axis=2)
highpassed = near_dft * np.repeat(highpass_filter_cut[:, :, np.newaxis], 3, axis=2)

# save 11,12
plt.imshow(calc_log_normalized_mag(lowpassed))
plt.show()
plt.imshow(calc_log_normalized_mag(highpassed))
plt.show()

# %% combine near and far

hybrid_dft = lowpassed + highpassed
# save 13
plt.imshow(calc_log_normalized_mag(hybrid_dft))
plt.show()


# %% spatial transform
def calc_idft(src):
    idft_full = list()
    for i in range(3):
        idft = np.fft.ifftshift(src[:, :, i])
        idft = np.fft.ifft2(idft)
        idft_full.append(idft)
    return np.stack(idft_full, axis=2).astype(float)


hybrid = calc_idft(hybrid_dft).astype(int)

# save 14,15
plt.imshow(hybrid)
plt.show()

# %% additional tests
pyr = np.ones((hybrid.shape[0], hybrid.shape[1] * 2, 3), dtype=np.float) * 255

ptr_i = pyr.shape[0] - 1
ptr_j = pyr.shape[1] // 2
ctm = hybrid.astype(np.float)
pyr[0:ctm.shape[0], 0:ctm.shape[1]] = ctm
for k in range(5):
    ctm = np.ascontiguousarray(ctm, dtype=np.float)
    ctm = np.array(ctm)
    ctm = cv.resize(ctm, (0, 0), ctm, 0.5, 0.5, interpolation=cv.INTER_LINEAR)
    ptr_i -= ctm.shape[0]
    pyr[ptr_i:ptr_i + ctm.shape[0], ptr_j:ptr_j + ctm.shape[1]] = ctm

pyr = np.clip(pyr, 0, 255).astype(np.uint8)
plt.imshow(pyr)
plt.show()
