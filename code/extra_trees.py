import numpy as np
from get_data import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation

data_set = getData()

trainX = data_set[:, :-1]
trainY = data_set[:, -1]

row , col = trainX.shape
# scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
# scaler.fit(trainX)
# trainX = scaler.transform(trainX)


extraTree = ExtraTreesClassifier(n_estimators = 50 , criterion = 'gini', max_features = None, n_jobs = -2, verbose = 3, class_weight = 'auto')

# all codes working

# using k folds

# kFold = cross_validation.KFold(n = row, n_folds = 5, shuffle = True)
# i = 0
# for train_index, test_index in kFold:
# 	trainXX = trainX[train_index]
# 	testXX = trainX[test_index]
# 	trainYY = trainY[train_index]
# 	testYY = trainY[test_index]
# 	extraTree.fit(trainXX, trainYY)
# 	predYY = extraTree.predict(testXX)
# 	i = i + 1
# 	print 'For %d fold the results are as follows:' %(i)
# 	print metrics.classification_report(testYY, predYY)
# 	print "\n"
	


#using cross_val_score

# cross_tech_scores = cross_validation.cross_val_score(extraTree, trainX, trainY, scoring = 'f1_macro', cv = 5, n_jobs = -2, verbose = 3)
# print cross_tech_scores.scores

# using stratified shuffled split

stratSplit = cross_validation.StratifiedKFold(trainY, n_folds = 5, shuffle = True)
i = 0
for train_index, test_index in stratSplit:
	trainXX = trainX[train_index]
	testXX = trainX[test_index]
	trainYY = trainY[train_index]
	testYY = trainY[test_index]
	extraTree.fit(trainXX, trainYY)
	predYY = extraTree.predict(testXX)
	i = i + 1
	print 'For %d fold the results are as follows:' %(i)
	print metrics.classification_report(testYY, predYY)
	print "\n"
	