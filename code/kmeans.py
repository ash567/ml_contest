import numpy as np 
import sklearn
import sklearn.cluster

trainX = np.genfromtxt('train_X.csv', delimiter = ',')
trainY = np.genfromtxt('train_Y.csv')

kmeans = sklearn.cluster.KMeans(n_clusters = 10)
kmeans.fit(trainX)
labels = kmeans.predict(trainX)
np.savetxt("cluster_labels.csv", labels, delimiter = ',', fmt = '%d')