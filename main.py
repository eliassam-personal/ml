import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi, e, sqrt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from random import shuffle


# f = np.random.normal(mu, sigma, 10)

mu = 0
sigma = 0.2

# unknowns
w0 = -1.2
w1 = 0.9

# plot the line
xlist = np.linspace(-1, 1, 200)
ylist = w0 + w1 * xlist
""" fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xlist, ylist, "g--") """


epsilon = np.random.normal(mu, sigma, 200)
Dtraining = [sum(x) for x in zip(ylist, epsilon)]


# ax.scatter(xlist, Dtraining, s=10, c="red")

phi = [[1, x] for x in xlist]

""" w = np.matmul(
    np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), np.transpose(phi)),
    Dtraining,
)


ylist = w[0] + w[1] * xlist
ax.plot(xlist, ylist, "r") """

alpha = 2
beta = sigma ** (-2)

N = 200

SnInverse = alpha * np.identity(2) + beta * np.matmul(np.transpose(phi), phi)
Sn = np.linalg.inv(SnInverse)
mn = beta * np.matmul(np.matmul(Sn, np.transpose(phi)), Dtraining)


print(mn)
print(Sn)

w0list = np.linspace(-3.0, 1.0, 200)
w1list = np.linspace(-2.0, 2.0, 200)
W0arr, W1arr = np.meshgrid(w0list, w1list)
pos = np.dstack((W0arr, W1arr))
rv = multivariate_normal(mn, Sn)
Wpriorpdf = rv.pdf(pos)
plt.contour(W0arr, W1arr, Wpriorpdf)


randomIndices = [i for i in range(200)]
shuffle(randomIndices)
sampleSize = 5
randomIndices = randomIndices[:sampleSize]
randomTuples = [(float(xlist[i]), float(Dtraining[i])) for i in randomIndices]

norm = lambda x, mu, sigma: (1 / (sigma * sqrt(2 * pi))) * e ** (
    (-((x - mu) ** 2)) / (2 * sigma**2)
)
p = lambda x, t, w0, w1: norm(t - w0 - w1 * x, mu, sigma)

# w0 + w1*x + epsilon = y
# epsilon = y - w0 - w1*x

Z = []

for i in range(200):
    Z.append([])
    w1 = W1arr[i][0]
    for j in range(200):
        w0 = W0arr[0][j]
        prob = map(
            p,
            [e[0] for e in randomTuples],
            [e[1] for e in randomTuples],
            [w0] * sampleSize,
            [w1] * sampleSize,
        )
        prob = list(prob)
        prob = np.prod(prob)
        Z[-1].append(prob)

print(randomTuples)

plt.contour(W0arr, W1arr, Z)

plt.xlabel("w0")
plt.ylabel("w1")


plt.show()
