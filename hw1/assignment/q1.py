# %% imports
import numpy as np
import cv2 as cv
import math


# %% define of transform functions

def log_transform(beta):
    return lambda x: np.uint8(round(abs(math.log(float(x) / 255 * (beta - 1) + 1) / math.log(beta) * 255)))


def optimize(src, beta):
    func = np.arange(256)
    func = np.vectorize(log_transform(beta))(func)
    return func[src]


# %% image loading
pic_orig = cv.imread('./data/Dark.jpg')

# %% apply transformation
pic_res = optimize(pic_orig, 25)

# %% remove salt and pepper noise
pic_res = cv.medianBlur(pic_res, 7)

# %% save image
cv.imwrite('./out/res01.jpg', pic_res)
