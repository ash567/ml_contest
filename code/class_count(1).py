fp = open('train_Y.csv', 'r')
count = [0 for i in range(100)]
for lines in fp:
	count[int(lines)] = count[int(lines)] + 1
count = [(count[i], i) for i in range(100)]
count.sort()
print count
