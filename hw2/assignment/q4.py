# %%
import cv2 as cv
import numpy as np
import math

near = cv.imread('./data/q4_01_near.jpg')
far = cv.imread('./data/q4_02_far.jpg')

# %% registering faces

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

cv.imwrite('./out/q4_03_near.jpg', near)
cv.imwrite('./out/q4_04_far.jpg', far)


# %% calculating dft
def calc_log_normalized_mag(src):
    src = np.log(np.abs(src) + 0.00000001)
    src -= src.min(initial=0)
    src /= src.max(initial=1)
    return (src * 255).astype(np.uint8)


def calc_dft(src):
    dft_full = list()
    for i in range(3):
        dft = np.fft.fft2(src[:, :, i])
        dft = np.fft.fftshift(dft)
        dft_full.append(dft)
    return np.stack(dft_full, axis=2)


near_dft = calc_dft(near)
far_dft = calc_dft(far)

cv.imwrite('./out/q4_05_dft_near.jpg', calc_log_normalized_mag(near_dft))
cv.imwrite('./out/q4_06_dft_far.jpg', calc_log_normalized_mag(far_dft))


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

cv.imwrite(f'./out/q4_07_highpass_{highpass_sigma}.jpg', (highpass_filter * 255).astype(np.uint8))
cv.imwrite(f'./out/q4_08_lowpass_{lowpass_sigma}.jpg', (lowpass_filter * 255).astype(np.uint8))


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

lowpass_cutoff_filter = cutoff_filter(lowpass_filter.shape, lowpass_cutoff)
highpass_cutoff_filter = (1 - cutoff_filter(highpass_filter.shape, highpass_cutoff))

lowpass_filter_cut = lowpass_cutoff_filter * lowpass_filter
highpass_filter_cut = highpass_cutoff_filter * highpass_filter

cv.imwrite('./out/q4_09_highpass_cutoff.jpg', (highpass_filter_cut * 255).astype(np.uint8))
cv.imwrite('./out/q4_10_lowpass_cutoff.jpg', (lowpass_filter_cut * 255).astype(np.uint8))

# %% apply filters

lowpassed = far_dft * np.repeat(lowpass_filter_cut[:, :, np.newaxis], 3, axis=2)
highpassed = near_dft * np.repeat(highpass_filter_cut[:, :, np.newaxis], 3, axis=2)

cv.imwrite('./out/q4_11_highpassed.jpg', calc_log_normalized_mag(highpassed))
cv.imwrite('./out/q4_12_lowpassed.jpg', calc_log_normalized_mag(lowpassed))

# %% combine near and far

hybrid_dft = lowpassed + highpassed
hybrid_dft *= np.repeat((1.5 - 0.5 * (highpass_cutoff_filter + lowpass_cutoff_filter))[:, :, np.newaxis], 3, axis=2)
cv.imwrite('./out/q4_13_hybrid_frequency.jpg', calc_log_normalized_mag(hybrid_dft))


# %% spatial transform
def calc_idft(src):
    idft_full = list()
    for i in range(3):
        idft = np.fft.ifftshift(src[:, :, i])
        idft = np.fft.ifft2(idft)
        idft_full.append(idft)
    return np.real(np.stack(idft_full, axis=2))


hybrid_near = np.clip(calc_idft(hybrid_dft), 0, 255).astype(np.uint8)
hybrid_far = hybrid_near.copy()
hybrid_far = cv.resize(hybrid_far, (0, 0), hybrid_far, 0.2, 0.2, cv.INTER_AREA)

cv.imwrite('./out/q4_14_hybrid_near.jpg', hybrid_near)
cv.imwrite('./out/q4_15_hybrid_far.jpg', hybrid_far)

# %% additional tests
hybrid_pyr = np.ones((hybrid_near.shape[0], int(hybrid_near.shape[1] * 1.5) + 1, 3), dtype=np.float) * 255

ptr_i = hybrid_pyr.shape[0] - 1
ptr_j = hybrid_near.shape[1]
ctm = hybrid_near.astype(np.float)
hybrid_pyr[0:ctm.shape[0], 0:ctm.shape[1]] = ctm
for k in range(5):
    ctm = np.ascontiguousarray(ctm, dtype=np.float)
    ctm = cv.resize(ctm, (0, 0), ctm, 0.5, 0.5, interpolation=cv.INTER_LINEAR)
    ptr_i -= ctm.shape[0]
    hybrid_pyr[ptr_i:ptr_i + ctm.shape[0], ptr_j:ptr_j + ctm.shape[1]] = ctm

hybrid_pyr = np.clip(hybrid_pyr, 0, 255).astype(np.uint8)
cv.imwrite('./out/q4_16_hybrid_pyramid.jpg', hybrid_pyr)
