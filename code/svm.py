from get_data import *
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.svm import NuSVC

# change the kernel accordingly

train = getData();

trainX = train[:, :-1]
trainY = train[:, -1]

scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))

clf = NuSVC(nu = 0.2, kernel = 'linear', class_weight = 'auto', cache_size = 2000, gamma = .01)
# val = cross_validation.cross_val_score(clf, trainX, trainY, scoring = 'f1_macro', cv = 5, n_jobs = -2, verbose = 3)


stratSplit = cross_validation.StratifiedKFold(trainY, n_folds = 5, shuffle = True)
i = 0
for train_index, test_index in stratSplit:
	trainXX = trainX[train_index]
	testXX = trainX[test_index]
	scaler.fit(trainXX)
	trainXX = scaler.transform(trainXX)
	testXX = scaler.transform(testXX)
	trainYY = trainY[train_index]
	testYY = trainY[test_index]
	clf.fit(trainXX, trainYY)
	predYY = clf.predict(testXX)
	i = i + 1
	print 'For %d fold the results are as follows:' %(i)
	print metrics.classification_report(testYY, predYY, digits = 10)
	print "\n"