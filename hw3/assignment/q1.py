from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans

f = open('./data/Points.txt')
n = int(f.readline())
points = np.array([list(map(float, f.readline().split())) for i in range(n)])

plt.scatter(points[:, 0], points[:, 1])
plt.savefig('./out/res01.jpg')
plt.close()


def to_polar(a):
    x = a[0]
    y = a[1]
    return math.sqrt(x ** 2 + y ** 2), math.atan(y / x)


lbl2 = KMeans(n_clusters=5).fit_predict(points)
plt.scatter(points[:, 0], points[:, 1], c=lbl2)
plt.savefig('./out/res02.jpg')
plt.close()

polar = np.apply_along_axis(to_polar, 1, points)
lbl = KMeans(n_clusters=2).fit_predict(polar)
plt.scatter(polar[:, 0], polar[:, 1], c=lbl)
plt.savefig('./out/res03.jpg')
plt.close()
