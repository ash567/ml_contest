from get_data import *
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import naive_bayes

# from sklearn.neighbors import NearestCentroid

from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation

distinct = 100
data = getData()
dataX = data[:, :-1]
dataY = data[:, -1]


# change the kernel accordingly

# clf = LabelPropagation()


# scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
# scaler.fit(trainX)
# trainX = scaler.transform(trainX)

# clf = naive_bayes.MultinomialNB(alpha = 0.001)

# clf = naive_bayes.BernoulliNB()
# clf = NearestCentroid()																																																																																																									
# val = cross_validation.cross_val_score(clf, trainX, trainY, scoring = 'f1_macro', cv = 5, n_jobs = -2, verbose = 3)




# let all the other classes be named as the 100
# let list of the classes to be trained on

bad_classes = [0, 4, 10, 27, 28, 31, 33, 34, 38, 40, 46, 54, 69, 70, 73, 77, 86, 88, 98]

# high = [dataY == 67]
# low = [dataY != 67]

# highData = data[high]
# lowData = data[low]

# count = np.zeros((distinct,1))
# for i in range(len(dataY)):
# 	count[int(dataY[i])] = count[int(dataY[i])] + 1

# class_count = []
# for i in range(len(count)):
# 	class_count.append(( int(count[i]), i))


# class_count.sort()


# for a in class_count:
# 	print a
# print class_count
# count.sort()
# print count
# print count.shape
# print sum(dataY == 100)


from sklearn import svm

for i in xrange(len(dataY)):
	if dataY[i] not in bad_classes:
		dataY[i] = 100

clf = svm.SVC(class_weight = 'auto', cache_size = 2000,  C =.01)


stratSplit = cross_validation.StratifiedKFold(dataY, n_folds = 5, shuffle = True)


i = 0
for train_index, test_index in stratSplit:
	trainXX = dataX[train_index]
	testXX = dataX[test_index]

	scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
	scaler.fit(trainXX)
	trainXX = scaler.transform(trainXX)
	testXX = scaler.transform(testXX)

	trainYY = dataY[train_index]
	testYY = dataY[test_index]

	clf.fit(trainXX, trainYY)
	predYY = clf.predict(testXX)
	i = i + 1
	print 'For %d fold the results are as follows:' %(i)
	print metrics.classification_report(testYY, predYY)
	print "\n"
