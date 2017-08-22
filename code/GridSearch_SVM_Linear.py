import numpy as np
from get_data import *
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import grid_search

# change the kernel accordingly

train = getData();

trainX = train[:, :-1]
trainY = train[:, -1]

scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
scaler.fit(trainX)
trainX = scaler.transform(trainX)

C_range = np.logspace(-3, 10, 20)
param_grid = dict(C=C_range, kernel = ['linear'])

# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)


cv = cross_validation.StratifiedShuffleSplit(trainY, n_iter=5, test_size=0.2, random_state=42)
grid = grid_search.GridSearchCV(SVC(), param_grid=param_grid, cv=cv, scoring = 'f1_macro', n_jobs = -2, verbose = 4)
grid.fit(trainX, trainY)

