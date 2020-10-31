# %% imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# %% load image
pic_orig = cv.cvtColor(cv.imread('./data/Dark.jpg'), cv.COLOR_BGR2RGB)
pic_tar = cv.cvtColor(cv.imread('./data/Pink.jpg'), cv.COLOR_BGR2RGB)


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

    return func[source]


# %%
pic_res = pic_orig.copy()
for i in range(3):
    print(f'channel {i + 1} of 3')
    pic_res[:, :, i] = specify_histogram(pic_orig[:, :, i], pic_tar[:, :, i])
plt.imsave('./out/res06.jpg', pic_res)

# %%
plt.figure(dpi=200)

color = ('r', 'g', 'b')
for j, col in enumerate(color):
    hist = cv.calcHist([pic_res], [j], None, [256], [0, 256]).astype(np.float64).ravel()
    hist /= hist.sum()
    plt.plot(hist, color=col, linewidth=1)
    plt.plot(np.nonzero(hist)[0], hist[np.nonzero(hist)],'--', color=col, linewidth=0.5)
    plt.xlim([0, 256])

plt.savefig('./out/res05.jpg')
