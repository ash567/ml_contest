fp = open('cluster_labels.csv', 'r')
ls = [0 for i in range(10)]
for lines in fp:
	ls[int(lines)] = ls[int(lines)] + 1
print ls