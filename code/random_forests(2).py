import numpy as np 
import sklearn.feature_selection
import sklearn.ensemble 
import sklearn.metrics
import os

numberOfFeatures = int(raw_input('How many features?'))
n_estimators = int(raw_input('How many random forests?'))

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
RFEstimator = sklearn.ensemble.RandomForestClassifier(n_jobs = -2, class_weight = 'auto', n_estimators = 100)
RFEstimator.fit(trainData[:, :-1], trainData[:, -1])
featureWeights = zip(RFEstimator.feature_importances_, range(2048))
featureWeights.sort()
featureWeights = [featureWeights[i][1] for i in range(2048)]
featureWeights = featureWeights[:2048-numberOfFeatures]
trainData = np.delete(trainData, featureWeights, axis = 1)
validationData = np.delete(validationData, featureWeights, axis = 1)
predY = RFEstimator.predict(validationData[:, :-1])
print sklearn.metrics.classification_report(validationData[:, -1], predY, labels=np.array([i for i in range(100)]), target_names=['Class %d' % (i) for i in range(100)], sample_weight=None, digits=3)