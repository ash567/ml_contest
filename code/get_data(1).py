import numpy as np 
import os

# Use trainData[:, :-1] as the features for training data.
# Use trainData[:, -1] as the labels for the training data.
# from get_data import *

def getData():
	if os.path.isfile('train.npy'):
		trainData = np.load('train.npy')
	else:
		trainX = np.genfromtxt('train_X.csv', delimiter = ',')
		trainY = np.genfromtxt('train_Y.csv')
		trainY = np.array([trainY])
		trainData = np.concatenate((trainX, trainY.T), axis = 1)
		np.save('train', trainData)
	return trainData