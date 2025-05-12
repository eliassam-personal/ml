import numpy as np
import matplotlib.pyplot as plt

t = lambda w0, w1, w2, x1, x2: w0 + w1 * x1**2 + w2 * x2**3
epsilon = lambda std: np.random.normal(0, std)
w = [0, 2.5, -0.5]
w0, w1, w2 = w

x1List = np.linspace(-1.0, 1.0, 41).tolist()
x2List = np.linspace(-1.0, 1.0, 41).tolist()

for i,std in enumerate([0, 0.5, 0.8, 1.2]):
    tMatrix = [
        [t(w0, w1, w2, x1, x2) + epsilon(std) for x2 in x2List] for x1 in x1List
    ]  # [(x1,x2)]
    plt.subplot(2, 2, i+1)
    plt.title(f"σ: {std}")
    if i % 2 == 0: plt.ylabel("x1")
    if i >= 2: plt.xlabel("x2")
    plt.contour(x1List, x2List, tMatrix, 15)

plt.show()
