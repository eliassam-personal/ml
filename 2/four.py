import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

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

trainingVector = [t for tRow in trainingDataMatrix for t in tRow]

phi = lambda x1, x2: [1, x1**2, x2**3]

design = [phi(x1, x2) for x1 in trainingDataX1List for x2 in trainingDataX2List]

wML = np.matmul(
    np.matmul(
        np.linalg.inv(np.matmul(np.transpose(design), design)), np.transpose(design)
    ),
    trainingVector,
)

bML = len(trainingVector) / sum(
    [
        (trainingVector[i] - np.matmul(design[i], wML)) ** 2
        for i in range(len(trainingVector))
    ]
)

testVector = [t for tRow in testDataMatrix for t in tRow]

testDesign = [phi(x1, x2) for x1 in testDataX1List for x2 in testDataX2List]

mse = sum(
    [
        (np.matmul(testDesign[i], wML) - testVector[i]) ** 2
        for i in range(len(testVector))
    ]
) / len(testVector)

variance = 1 / bML

alphaList = [0.3, 0.7, 2.0]

alpha = alphaList[0]

SnInverse = alpha * np.identity(3) + bML * np.matmul(np.transpose(design), design)
Sn = np.linalg.inv(SnInverse)
mn = bML * np.matmul(np.matmul(Sn, np.transpose(design)), trainingVector)

bayesianVariance = lambda x1, x2: (1 / bML) + np.matmul(
    np.matmul(phi(x1, x2), Sn), np.transpose(phi(x1, x2))
)


posterior = lambda x1, x2: (
    float(np.matmul(np.transpose(mn), phi(x1, x2))),
    float(bayesianVariance(x1, x2)),
)

posteriorMeans = testDesign @ mn
posteriorVariances = (1 / bML) + np.diag(
    np.matmul(np.matmul(testDesign, Sn), np.transpose(testDesign))
)


posteriors = [
    (float(posteriorMeans[i]), float(posteriorVariances[i]))
    for i in range(len(posteriorMeans))
]

for x1 in testDataX1List:
    for x2 
    print(p)


# bayesianMean = (bML * np.matmul(Sn, ))
