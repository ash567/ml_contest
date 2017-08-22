import numpy as np
from get_data import *
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.linear_model import PassiveAggressiveClassifier

# change the kernel accordingly

train = getData();
trainX = train[:, :-1]
trainY = train[:, -1]

clf = PassiveAggressiveClassifier()

print clf.get_params().keys()

C_range = np.logspace(-4, -2, 5)
param_grid = dict(C=C_range, n_iter = [5,10], loss = ['hinge', 'squared_hinge'], class_weight = ["balanced", None])

cv = cross_validation.StratifiedShuffleSplit(trainY, n_iter=4, test_size=0.2, random_state=42)
grid = grid_search.GridSearchCV(clf, param_grid=param_grid, cv=cv, scoring = 'f1_macro', n_jobs = -2, verbose = 4)
grid.fit(trainX, trainY)