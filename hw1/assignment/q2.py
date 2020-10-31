# %% imports
import numpy as np
import cv2 as cv

# %% load images
pic_yellow = cv.cvtColor(cv.imread('./data/Yellow.jpg'), cv.COLOR_BGR2HSV)
pic_pink = cv.cvtColor(cv.imread('./data/Pink.jpg'), cv.COLOR_BGR2HSV)

# %% define hue range conversions
yellow_hsv_range = (38. / 2, 54. / 2)  # hsv (40, 56) degree
red_hsv_range = (345. / 2, 355. / 2)  # hsv (345, 355) degree
pink1_hsv_range = (290. / 2, 360. / 2)  # hsv (325, 350) degree
pink2_hsv_range = (0. / 2, 25. / 2)
blue1_hsv_range = (215. / 2, 230. / 2)  # hsv (215,230) degree
blue2_hsv_range = (230. / 2, 235. / 2)


def linear_mapping(x, r1, r2):
    x = float(x)
    return (r2[1] - r2[0]) / (r1[1] - r1[0]) * (x - r1[0]) + r2[0]


def yellow2red(x):
    return linear_mapping(x, yellow_hsv_range, red_hsv_range)


def pink2blue(x):
    if pink1_hsv_range[0] <= x <= pink1_hsv_range[1]:
        return linear_mapping(x, pink1_hsv_range, blue1_hsv_range)
    if pink2_hsv_range[0] <= x <= pink2_hsv_range[1]:
        return linear_mapping(x, pink2_hsv_range, blue2_hsv_range)
    return x


def apply_transform(arr, transform):
    func = np.arange(256)
    func = np.vectorize(transform)(func)
    return func[arr]


# %% yellow to red
pic_all_red = pic_yellow.copy()

yellow_s_fade = 64. / 2
yellow_s_threshold = 0.35

pic_all_red[:, :, 0] = apply_transform(pic_all_red[:, :, 0], yellow2red)


# creating mask
def yellow_condition(p):
    h = p[0]
    s = p[1]
    if yellow_hsv_range[0] <= h <= yellow_hsv_range[1]:
        if 100 <= s <= 255:
            return np.uint8(255)
    if 54. / 2 < h:
        m = linear_mapping(h, (54. / 2, 64. / 2), (255, 0))
        if m * yellow_s_threshold <= s <= m:
            return np.uint8(255)
    return np.uint8(0)


yellow_mask = np.apply_along_axis(yellow_condition, 2, pic_yellow)

yellow_mask = cv.GaussianBlur(yellow_mask, (9, 9), 1.4)
yellow_mask = cv.threshold(yellow_mask, 180, 255, cv.THRESH_BINARY)[1]

pic_red_flower_only = cv.bitwise_and(pic_all_red, pic_all_red, mask=yellow_mask)
pic_yellow_no_flower = cv.bitwise_and(pic_yellow, pic_yellow, mask=cv.bitwise_not(yellow_mask))
pic_red = cv.add(pic_red_flower_only, pic_yellow_no_flower)

pic_red = cv.cvtColor(pic_red, cv.COLOR_HSV2BGR)
cv.imwrite('./out/res02.jpg', pic_red)

# %% pink to blue
pic_blue = pic_pink.copy()
pic_blue[:, :, 0] = apply_transform(pic_blue[:, :, 0], pink2blue)
pic_blue = cv.cvtColor(pic_blue, cv.COLOR_HSV2BGR)
cv.imwrite('./out/res03.jpg', pic_blue)
