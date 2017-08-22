import numpy as np 

trainX = np.genfromtxt('train_X.csv', delimiter = ',')
trainY = np.genfromtxt('train_Y.csv')
np.save('train_X', trainX)
np.save('train_Y', trainY)
