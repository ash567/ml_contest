import numpy as np 
from get_data import *
import sklearn.linear_model
import sklearn.cluster
import sklearn.metrics
import sklearn.ensemble
import sklearn.tree

train = getData()
trainX, trainY = train[:, :-1], train[:, -1]
testX = np.genfromtxt('test_X.csv', delimiter = ',')

gradTrees = sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, max_depth=3, max_features='log', verbose = 1)
pac = sklearn.linear_model.PassiveAggressiveClassifier(C=0.0001, n_iter=10, loss='squared_hinge', n_jobs=-1, class_weight='balanced')
log = sklearn.linear_model.LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', n_jobs=-1)
sgd = sklearn.linear_model.SGDClassifier(loss='hinge', alpha=0.08, fit_intercept=True, n_iter=5, n_jobs=-1, learning_rate='optimal', power_t=0.5, class_weight='balanced')
extraTrees = sklearn.ensemble.ExtraTreesClassifier(n_estimators=300, criterion='gini', max_features='auto', class_weight='balanced')
decisionTree = sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, class_weight='balanced')

eclf = sklearn.ensemble.VotingClassifier(estimators = [('pac', pac), ('sgd', sgd), ('log', log), ('extraTrees', extraTrees)], voting = 'hard')
eclf.fit(trainX, trainY)
pred = eclf.predict(testX)
