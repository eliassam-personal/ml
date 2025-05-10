import numpy as np
import matplotlib.pyplot as plt

std = 0.3

t = lambda w0, w1, w2, x1, x2: w0 + w1 * x1**2 + w2 * x2**3
epsilon = lambda: np.random.normal(0, std)
w = [0, 2.5, -0.5]
w0, w1, w2 = w

x1List = np.linspace(-1.0, 1.0, 41).tolist()
x2List = np.linspace(-1.0, 1.0, 41).tolist()

tMatrix = [
    [t(w0, w1, w2, x1, x2) + epsilon() for x2 in x2List] for x1 in x1List
]  # [(x1,x2)]

plt.figure(figsize=[5.0, 5.0])
plt.contour(x1List, x2List, tMatrix, 20)
plt.show()
