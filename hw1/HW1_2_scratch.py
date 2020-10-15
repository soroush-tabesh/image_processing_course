# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# %%
res_ratio = 0.25
dpi = 300


def show_rgb(pic):
    plt.figure(dpi=dpi)
    plt.imshow(pic)
    plt.show()


def show_hsv(pic):
    show_rgb(cv.cvtColor(pic, cv.COLOR_HSV2RGB))


# %%

pic_yellow = cv.cvtColor(cv.imread('./data/Yellow.jpg'), cv.COLOR_BGR2HSV)
pic_yellow = cv.resize(pic_yellow, (0, 0), pic_yellow, res_ratio, res_ratio, interpolation=cv.INTER_AREA)
show_hsv(pic_yellow)

# %%

pic_pink = cv.cvtColor(cv.imread('./data/Pink.jpg'), cv.COLOR_BGR2HSV)
pic_pink = cv.resize(pic_pink, (0, 0), pic_pink, res_ratio, res_ratio, interpolation=cv.INTER_AREA)
show_hsv(pic_pink)

# %%
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
    if yellow_hsv_range[0] <= x <= yellow_hsv_range[1]:
        return linear_mapping(x, yellow_hsv_range, red_hsv_range)
    return x


def pink2blue(x):
    if pink1_hsv_range[0] <= x <= pink1_hsv_range[1]:
        return linear_mapping(x, pink1_hsv_range, blue1_hsv_range)
    if pink2_hsv_range[0] <= x <= pink2_hsv_range[1]:
        return linear_mapping(x, pink2_hsv_range, blue2_hsv_range)
    return x


# %% yellow 1
pic_yellow2red = pic_yellow.copy()
pic_yellow2red[:, :, 0] = np.vectorize(yellow2red)(pic_yellow2red[:, :, 0])
show_hsv(pic_yellow2red)

# %% yellow 2

pic_yellow2red = pic_yellow.copy()

yellow_s_fade = 64. / 2
yellow_s_threshold = 0.35
mpr1 = lambda x: linear_mapping(x, yellow_hsv_range, red_hsv_range)
mpr2 = lambda x: linear_mapping(x, (54. / 2, 64. / 2), (255, 0))

for i in np.ndindex(pic_yellow2red.shape[:2]):
    h = pic_yellow2red[i][0]
    s = pic_yellow2red[i][1]

    if yellow_hsv_range[0] <= h <= yellow_hsv_range[1]:
        if 100 <= s <= 255:
            h = mpr1(h)

    if 54. / 2 < h:
        m = mpr2(h)
        if m * yellow_s_threshold <= s <= m:
            h = mpr1(h)

    pic_yellow2red[i][0] = h
    pic_yellow2red[i][1] = s

show_hsv(pic_yellow2red)

# %% yellow 3
pic_red = pic_yellow.copy()
pic_yellow_copy = pic_yellow.copy()

yellow_s_fade = 64. / 2
yellow_s_threshold = 0.35
mpr1 = lambda x: linear_mapping(x, yellow_hsv_range, red_hsv_range)
mpr2 = lambda x: linear_mapping(x, (54. / 2, 64. / 2), (255, 0))

pic_red[:, :, 0] = np.vectorize(mpr1)(pic_red[:, :, 0])


# creating mask
def yellow_condition(p):
    h = p[0]
    s = p[1]
    if yellow_hsv_range[0] <= h <= yellow_hsv_range[1]:
        if 100 <= s <= 255:
            return np.uint8(255)
    if 54. / 2 < h:
        m = mpr2(h)
        if m * yellow_s_threshold <= s <= m:
            return np.uint8(255)
    return np.uint8(0)


yellow_mask = np.apply_along_axis(yellow_condition, 2, pic_yellow)
yellow_mask = cv.GaussianBlur(yellow_mask, (9, 9), 1)
yellow_mask = cv.threshold(yellow_mask, 180, 255, cv.THRESH_BINARY)[1]
pic_red = cv.bitwise_and(pic_red, pic_red, mask=yellow_mask)
pic_yellow_copy = cv.bitwise_and(pic_yellow_copy, pic_yellow_copy, mask=cv.bitwise_not(yellow_mask))
pic_yellow2red = cv.add(pic_red, pic_yellow_copy)

show_hsv(pic_yellow2red)

# %% pink 1
pic_pink2blue = pic_pink.copy()
pic_pink2blue[:, :, 0] = np.vectorize(pink2blue)(pic_pink2blue[:, :, 0])
show_hsv(pic_pink2blue)
