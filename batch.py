from cdet import *
import time
import scipy.spatial.distance as d

filenames = ['upright1.jpg']  #'flat1.jpg', 'flat2.jpg', 'flat3.jpg', 'upright1.jpg', 'upright2.jpg']

dentotal_flat = []
dentotal_up = []
i = 0
for filename in filenames:
	a = preproc(filename, "dilated2.png")

	drawing, centers = circle_detection("dilated2.png", "dilated.png", "circle.png")

	depth, width, clmns =  drawing.shape

	print depth, width


	depth_sort = qsort(centers, 0) 

	# start = time.time()
	c = d.pdist(depth_sort)
	# end = time.time()	

	# print end-start
	i+=1
	den = density(depth_sort, depth, width, 10, 10)
	if i<4:
		dentotal_flat.append(den)
	else: dentotal_up.append(den);
	print den
	# plt.close()
	# hist(den, 20, 'Histogram showing density distribution of '+ filename, 'Density of ball bearings', 'Frequency')

# plt.close()
# hist(dentotal_flat, 20, 'Histogram showing density distribution of all flat images', 'Density of ball bearings', 'Frequency')
# plt.close()
# hist(dentotal_up, 20, 'Histogram showing density distribution of all upright images', 'Density of ball bearings', 'Frequency')

# print len(centers)
# print sum(den)

# print sum(sorted(den)[15:]), len(sorted(den)[15:])
# print float(sum(sorted(den)[15:]))/float(len(sorted(den)[15:]))
# avg = []
# for i in range(1, len(den)):
# 	avg.append(sum(den[:i])/float(i))

# plt.plot(avg)
# plt.show(avg)

# c = mean(b, 10, 5) # Pixel mean to find average pixel density

# print(min(c), max(c))

hist(c, 50, 'Histogram showing the pair-distribution of all ball bearings', 'Distance between ball bearings', 'Frequency') # Histogram representation for average pixel density
