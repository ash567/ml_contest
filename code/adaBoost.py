from get_data import *
import numpy as np
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

(train, test)= getData();

# Not very good results - train avg f1 - .53, test avg f1 = .54 
# decTree = DecisionTreeClassifier(criterion = "gini", splitter = "random", max_features = "auto", max_depth = 3)
# bdt_real = AdaBoostClassifier(decTree, n_estimators=2000, learning_rate=1)

decTree = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_features = "auto", max_depth = 3)
bdt_real = AdaBoostClassifier(decTree, n_estimators=2000, learning_rate=.5)

bdt_real.fit(train[:, :-1], train[:, -1])
trainPredY = bdt_real.predict(train[:, :-1]);
print sklearn.metrics.classification_report(train[:, -1], trainPredY)

testPredY = bdt_real.predict(test[:, :-1])
print sklearn.metrics.classification_report(test[:, -1], testPredY)