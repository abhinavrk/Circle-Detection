from cdet import *
import time
import scipy.spatial.distance as d
import os

'''
Used to compare images to simulation (also on GitHub)

'''

a = os.listdir(path) #OS path - you should define this to be the current directory
filenames = []
for name in a:
	if '.jpg' in name:
		filenames.append(name)
details  =[]
entropyy = open("entropy.txt", 'a')

for filename in filenames:
	name = filename.split('.')[0]
	a = preproc(filename, "dilated2.png")
	drawing, centers, radius = circle_detection("dilated2.png",
		"dilated.png", "circle.png")

	#Centers normalized against radius - this is needed since for the simulation 
	#radius is automatically 1. 
	depth, width, clmns =  drawing.shape
	depth /= float(radius)
	width /= float(radius)

	#X-sorted centers
	depth_sort = qsort(centers, 0) 
	
	#Y-sorted centers
	width_sort = qsort(centers, 1)
	
	#P-distribution
	c = d.pdist(depth_sort)
	print name
	
	#Net Total Density Distribution in x and y
	denxy = density(depth_sort, depth, width, 10, 10)
	
	#Profiles in x and y direction
	depthsx, profilex = profile(depth_sort, depth, 0)
	depthsy, profiley = profile(width_sort, width, 1)
	
	#Entropy of system
	rand_dist, entropy1 = entropy(depth_sort)
	
	#Plot moving density
	hist(denxy, 20, 'Histogram showing the density \
		distribution of ball bearings', 'Density', 'Frequency')
	plt.savefig(name+'denxy.png')
	plt.close()
	
	#Start plotting everything else
	plt.subplot(2,1,1)
	plt.plot(depthsx, profilex)
	plt.title('X-Profile with entropy: {0}'.format(entropy1))
	plt.xlabel('x (in units radius)')
	plt.ylabel('Frequency')
	plt.subplot(2,1,2)
	plt.title('Y-Profile')
	plt.xlabel('y (in units radius)')
	plt.ylabel('Frequency')
	plt.plot(depthsy, profiley)
	plt.savefig(name+"profiles.png")
	plt.close()
	print entropy1
	
	#Store entropy data somewhere
	details.append([name, entropy1])
	
	#plot l-dist
	hist(c, 50, 'Histogram showing the l-dist '+
		'of ball bearings', '$l$ (in units radius)', 'Frequency')
	plt.savefig(name+'l-dist.png')
	plt.close()
	np.savetxt(name+'_centers.txt', centers)

#write entropy data to a file
for x in details:
	entropyy.write(x[0])
	entropyy.write(',')
	entropyy.write(str(x[1]))
	entropyy.write('\n')
entropyy.close()
