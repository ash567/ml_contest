from get_data import *
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn import grid_search

from sklearn import metrics


train = getData();
trainX = train[:, :-1]
trainY = train[:, -1]

cv = cross_validation.StratifiedShuffleSplit(trainY, n_iter=5, test_size=0.2, random_state=42)
alpha_range = np.logspace(-5, 3, 40)
param_grid = dict(alpha=alpha_range, loss = ['modified_huber', 'hinge'], class_weight = ['auto'], penalty = ['l1', 'l2'])

clf = SGDClassifier()

grid = grid_search.GridSearchCV(clf, param_grid=param_grid, cv=cv, scoring = 'f1_macro', n_jobs = -3, verbose = 4)
grid.fit(trainX, trainY)