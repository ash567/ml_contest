import numpy as np 
import sklearn.ensemble
import os

# Use trainData[:, :-1] as the features for training data.
# Use trainData[:, -1] as the labels for the training data.
# Use validationData[:, :-1] as the features for the validation data.
# Use validationData[:, -1] as the labels for the validation data.
# from get_data import *

def getData(numberOfFeatures = 2048):
	if os.path.isfile('train_X.npy') and os.path.isfile('train_Y.npy'):
		trainX = np.load('train_X.npy')
		trainY = np.load('train_Y.npy')
	else:
		trainX = np.genfromtxt('train_X.csv', delimiter = ',')
		trainY = np.genfromtxt('train_Y.csv')
		np.save('train_X', trainX)
		np.save('train_Y', trainY)
	trainY = np.array([trainY])
	train = np.concatenate((trainX, trainY.T), axis = 1)
	np.random.shuffle(train)
	trainSize = train.shape[0]
	trainData, validationData = train[:(trainSize*3)//4], train[(trainSize*3)//4:]
	if numberOfFeatures < 2048:
		RFEstimator = sklearn.ensemble.RandomForestClassifier(n_jobs = -2, class_weight = 'auto', n_estimators = 100)
		RFEstimator.fit(trainData[:, :-1], trainData[:, -1])
		featureWeights = zip(RFEstimator.feature_importances_, range(2048))
		featureWeights.sort()
		featureWeights = [featureWeights[i][1] for i in range(2048)]
		featureWeights = featureWeights[:2048-numberOfFeatures]
		trainData = np.delete(trainData, featureWeights, axis = 1)
		validationData = np.delete(validationData, featureWeights, axis = 1)
	return (trainData, validationData)