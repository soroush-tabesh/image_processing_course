# %%
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans

f = open('./data/Points.txt')
n = int(f.readline())
points = np.array([list(map(float, f.readline().split())) for i in range(n)])

plt.scatter(points[:, 0], points[:, 1])
plt.show()


# %%
def to_polar(a):
    x = a[0]
    y = a[1]
    return math.sqrt(x ** 2 + y ** 2), math.atan(y / x)


polar = np.apply_along_axis(to_polar, 1, points)
plt.scatter(polar[:, 0], polar[:, 1])
plt.show()
# %%

lbl = KMeans(n_clusters=2).fit_predict(polar)
plt.scatter(polar[:, 0], polar[:, 1], c=lbl)
plt.show()
#%%

lbl2 = KMeans(n_clusters=2).fit_predict(points)
plt.scatter(points[:, 0], points[:, 1], c=lbl2)
plt.show()
