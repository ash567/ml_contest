import numpy as np 
import sklearn.feature_selection
import sklearn.ensemble 
import sklearn.metrics
from get_data import *

numberOfFeatures = int(raw_input('How many features?'))
n_estimators = int(raw_input('How many random forests?'))

trainData, validationData = getData(numberOfFeatures)
RFEstimator = sklearn.ensemble.RandomForestClassifier(n_jobs = -2, class_weight = 'auto', n_estimators = n_estimators)
RFEstimator.fit(trainData[:, :-1], trainData[:, -1])
predY = RFEstimator.predict(validationData[:, :-1])
print sklearn.metrics.classification_report(validationData[:, -1], predY, labels=np.array([i for i in range(100)]), target_names=['Class %d' % (i) for i in range(100)], sample_weight=None, digits=3)