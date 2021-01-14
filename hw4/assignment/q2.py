from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import math
import dlib


def tellme(s, plot):
    print(s)
    plot.set_title(s, fontsize=16)
    plt.draw()


def get_input_point(plot):
    init = []
    while True:
        tellme('press esc to continue', plot=plot)
        init = np.asarray(plt.ginput(-1, timeout=-1))
        ph = plot.scatter(init[:, 0], init[:, 1], c='b', s=1)
        tellme('Happy? esc for yes, mouse click for no', plot=plot)
        if plt.waitforbuttonpress():
            break
        ph.remove()
    tellme('Done', plot)
    # plt.close()
    return np.array(init, dtype=int)


lib_num = 81
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f'./data/shape_predictor_{lib_num}_face_landmarks.dat')
# https://github.com/codeniko/shape_predictor_81_face_landmarks/blob/master/shape_predictor_81_face_landmarks.dat

img_src_o = cv.imread('./data/images/elon-musk.jpg')
img_tar_o = cv.imread('./data/images/jack-ma.jpg')


def transform_fit_face(src, tar):
    src = src.copy()
    tar = tar.copy()

    rect_src = detector(src, 1)[0]
    rect_tar = detector(tar, 1)[0]

    area_src = rect_src.area()
    area_tar = rect_tar.area()
    resize_ratio = math.sqrt(area_tar / area_src)

    src = cv.resize(src, None, src, resize_ratio, resize_ratio, cv.INTER_AREA)

    rect_src = detector(src, 1)[0]
    rect_tar = detector(tar, 1)[0]

    mid_min_x = min(rect_src.center().x, rect_tar.center().x)
    mid_min_y = min(rect_src.center().y, rect_tar.center().y)

    src = src[(rect_src.center().y - mid_min_y):, (rect_src.center().x - mid_min_x):]
    tar = tar[(rect_tar.center().y - mid_min_y):, (rect_tar.center().x - mid_min_x):]

    f_shape = np.minimum(np.array(src.shape), np.array(tar.shape))

    src = src[:f_shape[0], :f_shape[1]]
    tar = tar[:f_shape[0], :f_shape[1]]

    return src, tar


img_src = img_src_o.copy()
img_tar = img_tar_o.copy()

img_src, img_tar = transform_fit_face(img_src, img_tar)


def generate_face_landmarks(src):
    rect = detector(src, 1)[0]
    landmarks = predictor(src, rect)

    landmarks_list = []
    for i in range(0, lib_num):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmarks_list.append((x, y))

    landmarks_list.extend(
        [(0, 0),
         (src.shape[1] - 1, 0),
         ((src.shape[1] - 1) // 2, 0),
         (0, src.shape[0] - 1),
         (0, (src.shape[0] - 1) // 2),
         ((src.shape[1] - 1) // 2, src.shape[0] - 1),
         (src.shape[1] - 1, src.shape[0] - 1),
         ((src.shape[1] - 1), (src.shape[0] - 1) // 2)]
    )

    return np.array(landmarks_list, dtype=int)


points_src = generate_face_landmarks(img_src)
points_tar = generate_face_landmarks(img_tar)

# input points
fig, axs = plt.subplots(ncols=2)
axs[0].imshow(cv.cvtColor(img_src, cv.COLOR_BGR2RGB))
axs[0].scatter(points_src[:, 0], points_src[:, 1], s=1, c='r')

axs[1].imshow(cv.cvtColor(img_tar, cv.COLOR_BGR2RGB))
axs[1].scatter(points_tar[:, 0], points_tar[:, 1], s=1, c='r')

points_src = np.concatenate((points_src, get_input_point(plot=axs[0])))
points_tar = np.concatenate((points_tar, get_input_point(plot=axs[1])))

plt.close()


def check_validity(vec, shape):
    if np.max(vec[0:][::2]) > shape[1] + 1:
        return False
    if np.max(vec[1:][::2]) > shape[0] + 1:
        return False
    if np.min(vec) < 0:
        return False
    return True


def triangulate(src, points):
    height, width = src.shape[:2]
    subdiv = cv.Subdiv2D((0, 0, width, height))

    points = [tuple(pt) for pt in points]
    index_of = {pt: i for i, pt in enumerate(points)}

    for pt in points:
        subdiv.insert(pt)

    triangles_list = subdiv.getTriangleList().astype(int)

    triangles_indexed = []
    for tri in triangles_list:
        if not check_validity(tri, src.shape):
            continue
        pts = []
        for i in range(3):
            pts.append(index_of[tuple(tri[2 * i:2 * i + 2])])
        triangles_indexed.append(tuple(pts))

    return np.array(triangles_indexed)


tries = triangulate(img_src, points_src)


def transform_triangle(src, tri_src, tri_tar, size):
    matrix = cv.getAffineTransform(np.array(tri_src, dtype=np.float32), np.array(tri_tar, dtype=np.float32))
    return cv.warpAffine(src, matrix, size, None, flags=cv.INTER_AREA, borderMode=cv.BORDER_REFLECT)


def interpolate_triangle(img_src, img_tar, dest, tri_src, tri_tar, tri_inter, alpha):
    tri_src = np.array(tri_src)
    tri_tar = np.array(tri_tar)
    tri_inter = np.array(tri_inter)

    bound_src = np.int32([tri_src[:, 0].min(), tri_src[:, 1].min(),
                          tri_src[:, 0].max() + 1, tri_src[:, 1].max() + 1])
    bound_tar = np.int32([tri_tar[:, 0].min(), tri_tar[:, 1].min(),
                          tri_tar[:, 0].max() + 1, tri_tar[:, 1].max() + 1])
    bound_inter = np.int32([tri_inter[:, 0].min(), tri_inter[:, 1].min(),
                            tri_inter[:, 0].max() + 1, tri_inter[:, 1].max() + 1])

    w = bound_inter[2] - bound_inter[0]
    h = bound_inter[3] - bound_inter[1]

    tri_src[:, 0] -= bound_src[0]
    tri_src[:, 1] -= bound_src[1]
    tri_tar[:, 0] -= bound_tar[0]
    tri_tar[:, 1] -= bound_tar[1]
    tri_inter[:, 0] -= bound_inter[0]
    tri_inter[:, 1] -= bound_inter[1]

    mask = np.zeros((h, w, 3))
    cv.fillConvexPoly(mask, tri_inter.astype(np.int32), (1.0, 1.0, 1.0), 16, 0)

    rect_src = img_src[bound_src[1]:bound_src[3], bound_src[0]:bound_src[2]]
    rect_tar = img_tar[bound_tar[1]:bound_tar[3], bound_tar[0]:bound_tar[2]]

    warped_src = transform_triangle(rect_src, tri_src, tri_inter, (w, h))
    warped_tar = transform_triangle(rect_tar, tri_tar, tri_inter, (w, h))

    warped_inter = (1.0 - alpha) * warped_src + alpha * warped_tar

    dest[bound_inter[1]:bound_inter[3],
    bound_inter[0]:bound_inter[2]] = dest[bound_inter[1]:bound_inter[3],
                                     bound_inter[0]:bound_inter[2]] * (1 - mask) + warped_inter * mask


duration = 3
frame_rate = 15
frame_count = duration * frame_rate
video_writer = cv.VideoWriter('./out/res2.avi', cv.VideoWriter_fourcc(*'MJPG'),
                              frame_rate,
                              (img_src.shape[1], img_src.shape[0]))

img_src_f = img_src.astype(np.float)
img_tar_f = img_tar.astype(np.float)

for j in range(0, frame_count):
    alpha = j / (frame_count - 1)
    points_inter = (1 - alpha) * points_src + alpha * points_tar
    img_res = np.zeros_like(img_src)
    for tri in tries:
        interpolate_triangle(img_src_f, img_tar_f, img_res, points_src[tri], points_tar[tri],
                             points_inter[tri], alpha)
    video_writer.write(img_res)

video_writer.release()
