import numpy as np 
import sklearn.feature_selection
import sklearn.ensemble 
import sklearn.metrics

numberOfFeatures = int(raw_input('How many features?'))

trainX = np.genfromtxt('train_X.csv', delimiter = ',')
trainY = np.genfromtxt('train_Y.csv')
trainY = np.array([trainY])

train = np.concatenate((trainX, trainY.T), axis = 1)
np.random.shuffle(train)

features = np.genfromtxt('feature_ranks')

features = features[:, -1]

features = features[:2048-numberOfFeatures]

train = np.delete(train, features, axis = 1)

trainSize = train.shape[0]
trainData, validationData = train[:(trainSize*3)//4], train[(trainSize*3)//4:]

RFEstimator = sklearn.ensemble.RandomForestClassifier(n_jobs = 3, class_weight = 'auto', n_estimators = 300)

RFEstimator.fit(trainData[:, :-1], trainData[:, -1])

# RFEstimator.score(validationData[:, :-1], validationData[:, -1])

predY = RFEstimator.predict(validationData[:, :-1])

print sklearn.metrics.classification_report(validationData[:, -1], predY, labels=np.array([i for i in range(100)]), target_names=['Class %d' % (i) for i in range(100)], sample_weight=None, digits=3)