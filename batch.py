from cdet import *
import time
import scipy.spatial.distance as d

filenames = ['upright1.jpg']  
dentotal_flat = []
dentotal_up = []
i = 0
for filename in filenames:
	a = preproc(filename, "dilated2.png")

	drawing, centers = circle_detection("dilated2.png", "dilated.png", "circle.png")

	depth, width, clmns =  drawing.shape

	print depth, width


	depth_sort = qsort(centers, 0) 

	c = d.pdist(depth_sort)

	i+=1
	den = density(depth_sort, depth, width, 10, 10)
	if i<4:
		dentotal_flat.append(den)
	else: dentotal_up.append(den);
	print den


hist(c, 50, 'Histogram showing the pair-distribution of all ball bearings', 'Distance between ball bearings', 'Frequency') # Histogram representation for average pixel density
