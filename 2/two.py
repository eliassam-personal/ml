import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

std = 0.3

t = lambda w0, w1, w2, x1, x2: w0 + w1 * x1**2 + w2 * x2**3
epsilon = lambda: np.random.normal(0, std)
w = [0, 2.5, -0.5]
w0, w1, w2 = w

x1List = np.linspace(-1.0, 1.0, 6).tolist()
x2List = np.linspace(-1.0, 1.0, 6).tolist()

tMatrix = [
    [t(w0, w1, w2, x1, x2) + epsilon() for x2 in x2List] for x1 in x1List
]  # [(x1,x2)]

testDataX1List = [x1 for x1 in x1List if abs(x1) > 0.31]
testDataX2List = [x2 for x2 in x2List if abs(x2) > 0.31]

testDataMatrix = [
    [
        tMatrix[i][j]
        for j in range(len(tMatrix))
        if abs(x1List[i]) > 0.31 and abs(x2List[j]) > 0.31
    ]
    for i in range(len(tMatrix))
]

trainingDataX1List = [x1 for x1 in x1List if abs(x1) <= 0.31]
trainingDataX2List = [x2 for x2 in x2List if abs(x2) <= 0.31]
trainingDataMatrix = []

print(testDataX1List)
print(testDataX2List)
print(testDataMatrix)

plt.figure(figsize=[5.0, 5.0])
plt.contour(testDataX1List, testDataX2List, testDataMatrix, 20)
plt.show()
