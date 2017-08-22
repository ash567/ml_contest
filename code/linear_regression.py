import numpy as np 
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.grid_search
from get_data import *

# import os
# import sklearn.feature_selection
# import sklearn.ensemble 

# numberOfFeatures = int(raw_input('How many features?'))
# n_estimators = int(raw_input('How many random forests?'))

# if os.path.isfile('train_X.npy') and os.path.isfile('train_Y.npy'):
# 	trainX = np.load('train_X.npy')
# 	trainY = np.load('train_Y.npy')
# else:
# 	trainX = np.genfromtxt('train_X.csv', delimiter = ',')
# 	trainY = np.genfromtxt('train_Y.csv')
# 	np.save('train_X', trainX)
# 	np.save('train_Y', trainY)

# trainY = np.array([trainY])
# train = np.concatenate((trainX, trainY.T), axis = 1)

# import csv
# with open('test.csv', 'w', newline='') as fp:
# 	a = csv.writer(fp, delimiter=',')
# 	a.writerows(trainData)

# RFEstimator = sklearn.ensemble.RandomForestClassifier(n_jobs = -2, class_weight = 'auto', n_estimators = 100)
# RFEstimator.fit(trainData[:, :-1], trainData[:, -1])
# featureWeights = zip(RFEstimator.feature_importances_, range(2048))
# featureWeights.sort()
# featureWeights = [featureWeights[i][1] for i in range(2048)]
# featureWeights = featureWeights[:2048-numberOfFeatures]
# trainData = np.delete(trainData, featureWeights, axis = 1)
# validationData = np.delete(validationData, featureWeights, axis = 1)
# predY = RFEstimator.predict(validationData[:, :-1])
# print sklearn.metrics.classification_report(validationData[:, -1], predY, labels=np.array([i for i in range(100)]), target_names=['Class %d' % (i) for i in range(100)], sample_weight=None, digits=3)

# regr = sklearn.linear_model.LinearRegression(fit_intercept = True)
# regr.fit(trainData[:, :-1], trainData[:,-1])
# predY = regr.predict(validationData[:, :-1])

# for i in range(100):
# 	trainLabel = trainData[:, -1]==i
# 	trainData_ = np.concatenate((trainData[:, :-1], trainData[:, -1]), axis = 1)
# logistic = sklearn.linear_model.LogisticRegression(C=1e5)
# logistic.fit(trainData[:, :-1], trainData[:,-1])

# logistic = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True)
# logistic.fit(trainData[:, :-1], trainData[:,-1])

train = getData()
trainSize = train.shape[0]
kf = sklearn.cross_validation.KFold(trainSize, n_folds=5)
i= 1
for trainIndex, testIndex in kf:
	i
	trainData, testData = train[trainIndex], train[testIndex]
	trainX, testX = trainData[:,:-1], testData[:,:-1]
	trainY, testY = trainData[:,-1], testData[:,-1]
	ss = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
	ss.fit(trainX)
	trainX_ = ss.transform(trainX)
	testX_  = ss.transform(testX)
	# parameters = {'alpha':[0.00005, 0.00008, 0.00010, 0.00012]}
	clf = sklearn.linear_model.SGDClassifier(alpha=0.00011,class_weight="auto",n_jobs=-3)
	# clf = sklearn.grid_search.GridSearchCV(sgd, parameters, scoring)
	# clf = sklearn.grid_search.GridSearchCV(sgd, parameters, scoring = 'f1_macro', n_jobs=-2, cv = sklearn.cross_validation.StratifiedKFold(trainY, 3), verbose=3)
	clf.fit(trainX_,trainY)
	pred = clf.predict(testX_)
	# print sklearn.metrics.classification_report(testY, pred)
	print sklearn.metrics.f1_score(testY, pred, average='micro')
	print sklearn.metrics.f1_score(testY, pred, average='macro')
	i = i+1

# np.random.shuffle(train)
# trainData_, validationData_ = train[:(trainSize*3)//4], train[(trainSize*3)//4:]
# trainData = trainData_[:,:-1]
# trainLabel = trainData_[:,-1]
# validationData = validationData_[:,:-1]
# validationLabel = validationData_[:,-1]

# sz = trainX.shape[0]
# indicator = np.zeros((sz,100))
# for i in range(sz):
# 	indicator[ii,trainY[ii]] = 1

# Z = np.dot(trainX,trainX.T)
# W = np.dot(np.linalg.eig(Z),trainX.T)


# 0.78
# 0.80
# 0.83
# 0.81
# 0.82
