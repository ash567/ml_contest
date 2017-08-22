import numpy as np 
import os

# Use trainData[:, :-1] as the features for training data.
# Use trainData[:, -1] as the labels for the training data.
# from get_data import *

def getTest():
	if os.path.isfile('test.npy'):
		testX = np.load('test.npy')
	else:
		testX = np.genfromtxt('test_X.csv', delimiter = ',')
		np.save('test', testX)
	return testX