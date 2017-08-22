import numpy as np
import sklearn
from sklearn.qda import QDA

trainX = np.genfromtxt('train_X.csv', delimiter = ',')
trainY = np.genfromtxt('train_Y.csv')

clf = QDA()
clf.fit(trainX, trainY)

print clf.score(trainX, trainY)

