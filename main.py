import pylab as pb
import numpy as np
import matplotlib . pyplot as plt
from math import pi
from scipy . stats import multivariate_normal
from scipy . spatial . distance import cdist


# f = np.random.normal(mu, sigma, 10)

mu = 0
sigma = 0.2

w0 = -1.2
w1 = 0.9

# plot the line
xlist = np.linspace(-1,1,200)
ylist = w0 + w1*xlist 
fig,ax = plt.subplots(figsize=(10,7))
ax.plot(xlist,ylist, "g--")


epsilon = np.random.normal(mu, sigma, 200)
Dtraining = [sum(x) for x in zip(ylist, epsilon)]

ax.scatter(xlist,Dtraining, s=10, c="red")

phi = [[1, x] for x in xlist]

w = np.matmul(
        np.matmul(np.linalg.inv(np.matmul(
            np.transpose(phi),phi)), 
            np.transpose(phi)), 
        Dtraining)


ylist = w[0] + w[1]*xlist 
ax.plot(xlist,ylist,"r")

alpha = sigma**(-2)
beta = sigma**(-2)

N = 200

SnInverse= alpha * np.identity(2) + beta*np.matmul(np.transpose(phi), phi)
mn = beta * np.matmul(np.matmul(np.linalg.inv(SnInverse), np.transpose(phi)), Dtraining)

print(mn)
print(SnInverse)



plt.show()

