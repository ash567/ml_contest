import numpy as np 
import sklearn.cluster
import os

# Use trainData[:, :-1] as the features for training data.
# Use trainData[:, -1] as the labels for the training data.
# from get_data import *

def getData(numFeatures = 2048):
	if os.path.isfile('train.npy'):
		trainData = np.load('train.npy')
	else:
		trainX = np.genfromtxt('train_X.csv', delimiter = ',')
		trainY = np.genfromtxt('train_Y.csv')
		trainY = np.array([trainY])
		trainData = np.concatenate((trainX, trainY.T), axis = 1)
		np.save('train', trainData)
	if numFeatures != 2048:
		featSelect = sklearn.cluster.FeatureAgglomeration(n_clusters=numFeatures, affinity='euclidean', linkage='ward')
		featSelect.fit(trainData[: ,:-1])
		trainX = featSelect.transform(trainData[:, :-1])
		trainY = trainData[:, -1]
		trainData = np.concatenate((trainX, np.array([trainY]).T), axis = 1)
	return trainData