import numpy as np
import matplotlib.pyplot as plt

std = 0.3

t = lambda w0, w1, w2, x1, x2: w0 + w1 * x1**2 + w2 * x2**3
epsilon = lambda: np.random.normal(0, std)
w = [0, 2.5, -0.5]
w0, w1, w2 = w

x1List = np.linspace(-1.0, 1.0, 41).tolist()
x2List = np.linspace(-1.0, 1.0, 41).tolist()

testDataX1List = [x1 for x1 in x1List if abs(x1) > 0.31]
testDataX2List = [x2 for x2 in x2List if abs(x2) > 0.31]
testDataMatrix = [
    [t(w0, w1, w2, x1, x2) + epsilon() + epsilon() for x2 in testDataX2List]
    for x1 in testDataX1List
]

trainingDataX1List = [x1 for x1 in x1List if abs(x1) <= 0.31]
trainingDataX2List = [x2 for x2 in x2List if abs(x2) <= 0.31]
trainingDataMatrix = [
    [t(w0, w1, w2, x1, x2) + epsilon() for x2 in trainingDataX2List]
    for x1 in trainingDataX1List
]


plt.figure(figsize=[5.0, 5.0])
plt.contour(trainingDataX1List, trainingDataX2List, trainingDataMatrix, 20)
plt.show()
