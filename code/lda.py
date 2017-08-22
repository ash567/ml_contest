from get_data import *
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.lda import LDA
# change the kernel accordingly

train = getData();
trainX = train[:, :-1]
trainY = train[:, -1]

clf = LDA(solver = 'svd')
val1 = cross_validation.cross_val_score(clf, trainX, trainY, scoring = 'f1_macro', cv = 5, n_jobs = -1, verbose = 3)
print val1

# clf = LDA(solver = 'lsqr', )
# val2 = cross_validation.cross_val_score(clf, trainX, trainY, scoring = 'f1_macro', cv = 5, n_jobs = -5, verbose = 3)
# print val2


# clf = LDA(solver = 'eigen', shrinkage = True)
# val3 = cross_validation.cross_val_score(clf, trainX, trainY, scoring = 'f1_macro', cv = 5, n_jobs = -5, verbose = 3)
# # print val3