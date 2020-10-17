# %% imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# %% load image
res_ratio = 0.25
pic_orig = cv.cvtColor(cv.imread('./data/Dark.jpg'), cv.COLOR_BGR2RGB)
pic_tar = cv.cvtColor(cv.imread('./data/Pink.jpg'), cv.COLOR_BGR2RGB)
pic_orig = cv.resize(pic_orig, (0, 0), pic_orig, res_ratio, res_ratio, interpolation=cv.INTER_AREA)
pic_tar = cv.resize(pic_tar, (0, 0), pic_tar, res_ratio, res_ratio, interpolation=cv.INTER_AREA)

plt.gray()
fig = plt.figure(dpi=200)
plts = fig.add_gridspec(1, 2).subplots()
plts[0].imshow(pic_orig)
plts[1].imshow(pic_tar)
plt.show()


# %%
def specify_histogram(source, target):
    source_vals, source_freq = np.unique(source.ravel(), return_counts=True)
    target_vals, target_freq = np.unique(target.ravel(), return_counts=True)

    source_freq = np.cumsum(source_freq).astype(np.float64)
    source_freq /= source_freq[-1]
    target_freq = np.cumsum(target_freq).astype(np.float64)
    target_freq /= target_freq[-1]

    func = np.zeros(256, np.float64)

    for i, j in zip(source_vals, source_freq):
        func[i] = j

    transform = interp1d(target_freq, target_vals, bounds_error=False, fill_value=(target_vals[0], target_vals[-1]))
    func = transform(func).astype(np.uint8)

    return np.vectorize(lambda x: func[x])(source)


# %%
res = pic_orig.copy()
for i in range(3):
    res[:, :, i] = specify_histogram(pic_orig[:, :, i], pic_tar[:, :, i])
plt.imshow(res)
plt.show()

