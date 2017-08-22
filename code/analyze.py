import numpy as np
for name in ["../data/logistic.txt"]:
	print name
	f = open(name, "r")

	f1_score = np.zeros((100, 1))
	# print f1_score
	for line in f:
		ls = line.split()
		ys = [float(a) for a in ls]
		f1_score[int(ys[0])] = f1_score[int(ys[0])] + ys[3]

	f1_score = f1_score/5
	threshold = 0.750
	lis = []
	for i in np.arange(100):
		if(f1_score[i] < threshold):
			print i
			lis.append((f1_score[i], i))

	lis.sort()
	print lis
	print "\n"