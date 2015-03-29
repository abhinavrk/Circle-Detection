import numpy as np

'''
Find the average measure-of-goodness values for Logistic, Boltzmann and Gumbel curves. 

Do so for cases with and without edge effects.
'''
f = file("Results.txt")
res = np.loadtxt(f, delimiter = ' ')

def avg(arr):
	return sum(arr)/float(len(arr))

print 'Boltz: ', avg(bolt),'Logistic: ', avg(loggy),'Gumbel: ', avg(gummy)

print 'Boltz_mod: ', avg(boltm),'Logistic_mod: ', avg(loggym),'Gumbel_mod: ', avg(gummym)


