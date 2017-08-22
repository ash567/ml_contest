import numpy as np 
import sklearn.feature_selection
import sklearn.ensemble 
import sklearn.metrics
import sklearn.cross_validation
from get_data import *

n_estimators = int(raw_input('How many random forests?'))
trainData = getData()
RFEstimator = sklearn.ensemble.RandomForestClassifier(n_jobs = -2, class_weight = 'auto', n_estimators = n_estimators, max_features = None)
scores = sklearn.cross_validation.cross_val_score(RFEstimator, trainData[:, :-1], y = trainData[:, -1], cv = 5, n_jobs = -2)
print scores
