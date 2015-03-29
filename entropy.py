import numpy as np
'''
Calculate difference in entropy between sparse and densely packed regions
'''
a = open("entropy_all.txt")
b = open("entropy_sparse.txt")

all_name = []
sparse_name = []
all_val = []
sparse_val = []
avg = []

for line in a:
	words = line.split(',')
	all_name.append(words[0])
	all_val.append(float(words[1]))

for line in b:
	words = line.split(',')
	sparse_name.append(words[0])
	sparse_val.append(float(words[1]))

for i in range(len(sparse_name)):
	for j in range(len(all_name)):
		if sparse_name[i] == all_name[j]:
			diff = sparse_val[i] - all_val[j]
			avg.append(diff)
# print avg
print 'average: ', sum(avg)/float(len(avg))
print 'std: ', np.std(avg)
