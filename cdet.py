import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
Helper functions for batch.py
Uses OpenCV 3.0.0
'''

def inverte(imagem, name):
	'''
	Performs a bitwise NOT to convert image to its negative. 

	It does exist in CV2 but I didn't trust the code in the beta version - it looked sloppy. 

	'''
	imagem = (255-imagem)
	cv2.imwrite(name, imagem)

def preproc(image_head, preprocessed_img):

	'''
	Takes in an image and coverts it to gray-scale and then thresholds the image. 

	It amplifies circles in the image making them easier for detection. 
	
	It does this by first performing noise removal using a simple unitary kernel and then 
	performing morphological transformations (erode and dilate) to close any and all partially filled circles.

	Example of use:
	-----------------------------

	preproc("2015-01-13 12.jpg", "dilated2.png")

	"2015-01-13 12.jpg" - file to be processed. 
	"dilated2.png" - processed image. 


	'''
	img = cv2.imread(image_head)

	# Image gray-scale and thresholding
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# ret, thresh = cv2.threshold(gray,15,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
										 cv2.THRESH_BINARY_INV, 91, 3)
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	opening = cv2.erode(opening,kernel,iterations=1)
	opening = cv2.dilate(opening,kernel,iterations=2)

	inverte(opening, preprocessed_img)
	return preprocessed_img

def circle_detection(image_head, dil_img, circle_img):
	'''
	Takes in a preprocessed image and searches for circles within the image. 

	Example:
	----------------

	circle_detection("dilated2.png", "dilated.png", "circle.png")
	'''

	#Mean fitness value - will use later under contour. 
	meanfit = 25

	# Another gray-scaling + thresholding NECESSARY! Need retval
	original = cv2.imread(image_head, cv2.IMREAD_GRAYSCALE)
	retval, image = cv2.threshold(original, 15, 255, cv2.THRESH_BINARY)

	# el is structuring element of specified (5,5) size. It's an ellipse inside a rectangle basically. 
	el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

	# Gaussian blur of the image to get rid of any remaining noise.
	# image = cv2.GaussianBlur(image,(5,5),0)

	# Bilateral blur should be even better:
	blur = cv2.bilateralFilter(image,5,0,10)

	# Alternatives:
	# image = cv2.medianBlur(image,3)

	# Dilate to get only the true circles - dilate too much and you'll loose some circles. 
	image = cv2.dilate(image, el, iterations=3)

	#Store the cool dilated image. Useful for density calculations later. 
	cv2.imwrite(dil_img, image)

	# Find all relevant contours - source image is modified herein
	bar, contours, hierarchy= cv2.findContours(
	    image,
	    cv2.RETR_LIST,
	    cv2.CHAIN_APPROX_SIMPLE
	)

	#-----------NOTES
	# cv2.cv.CV_RETR_LIST == no need for any hierarchical relationships... ain't got no depth.
	# cv2.cv.CV_CHAIN_APPROX_SIMPLE == Don't need entire contour since we're going to model as an 
	# ellipse inside a rectangle so just give us 4 corners of the boundary rectangle

	# We're going to superimpose found contours and image and since
	# findContours()  alters input image we should probably 'reset' the image
	drawing = cv2.imread(image_head)

	# For Diagnostics:
	# mean_vals = []
	# reduced_mean = []
	# outliers = []
	centers = []
	radii = []
	for contour in contours:
	    area = cv2.contourArea(contour)
	    
	    # there is one contour that contains all others, filter it out
	    if area > 500:
	        continue

	    # Calculates the up-right bounding rectangle of the 4-point contour set (see findCountours() above)
	    br = cv2.boundingRect(contour)
	    radii.append(br[2])

	    #Find a mask for the boundary rectangle
	    mask = np.zeros(image.shape,np.uint8)
	    cv2.drawContours(mask,[contour],0,255,-1)

	    #Find mean value of pixels inside rectangle. 
	    mean_val = cv2.mean(image,mask = mask)
	    
	    #If the mean value too bright - then contours extend to cover white-space... thus bad contours 
	    #...they're not covering circle well... too large or not at all circle. 

	    #Told you we'd use meanfit ... it basically says: "Normal cells have a mean of <18... 
	    #anything more than 18 has whitespace ... == shitty contour"
	    if mean_val[0]>meanfit:
	    	# For Diagnostics
	        # outliers.append(mean_val[0])
	    	continue

	    #Again for Diagnostics:
	    #reduced_mean.append(mean_val[0]) 

	    # Think like moment of inertia -  Simple properties of the image which are found via image 
	    # moments include area (or total intensity), its centroid, and information about its orientation.
	    m = cv2.moments(contour)
	    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
	    centers.append(center)

	print("There are {} circles".format(len(centers)))
	# Once more for Diagnostics:
	# print min(mean_vals)
	# print max(reduced_mean)
	# print outliers

	# We were working with dilated circles, so actual radius has a multiplicative factor... 
	# but +5 works just as well. 
	radius = int(np.average(radii))

	# Draw center and circumference of circles onto the drawing.
	for center in centers:
	    cv2.circle(drawing, center, 3, (255, 0, 0), -1)
	    cv2.circle(drawing, center, radius, (0, 255, 0), 1)

	cv2.imwrite(circle_img, drawing)

	#Normalize centers to have radius = 1
	centers = np.array(centers)/float(radius)
	return drawing, centers, radius

def mean(image, size, step):
	'''
	Never used - here for completeness
	pixel averaging via mean. 
	'''
	#Find a mask for the boundary rectangle
	# width, height = cv2.GetSize(image)
	height, width = image.shape[:2]
	mean = []
	x2 = size
	y2 = size

	while(x2<width):
		while(y2<height):
			mask = cv2.rectangle(image, (x2-size, y2-size), (x2,y2), 0)
			value = cv2.mean(mask)[0]
			mean.append(value)
			y2+= step
		x2+= step
		y2 = size
	return mean

def hist(array, block, title, xlab, ylab):
	'''
	as in helper.py in checkin2 - gives histogram of array
	'''
	hist, bins = np.histogram(array, bins=block)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.title(title)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	# plt.show()
	return plt.bar(center, hist, align='center', width=width)

def qsort(a, i):
	'''
	Sorts data
	'''
    return sorted(a, key = lambda arr: arr[i]) 

def search(a, pos, value_start, value_end):
	'''
	Searches data
	'''
	if len(a)<1:
		return []

	empty = []
	i = 0
	for x in a:
		if x[pos] < value_end and x[pos]>=value_start:
			empty.append(x)
			i+=1
		elif x[pos]<value_start:
			i+=1
		else:
			return empty
	return empty

def density(arr, depth, width, points_d, points_w):
	'''
	density as in helper.py in checkin2
	'''
	# arr is a depth sorted array
	# depth, width are the image dimension
	# points_d, points_w are the number of points across depth and width.
	density = []
	depths = np.linspace(0, depth, points_d, endpoint = True)
	depths = depths.astype(int)
	widths = np.linspace(0, width, points_w,  endpoint = True)
	widths = widths.astype(int)

	for i in range(len(depths)-1):
		a = search(arr, 0, depths[i], depths[i+1])
		b = qsort(a,1)
		
		for j in range(len(widths)-1):
			c = search(b, 1, widths[j], widths[j+1])
			density.append(len(c))
	return density

def entropy(centers):
	'''
	entropy as in helper.py in checkin2
	'''
	dist = []
	N = len(centers)
	for i in range(N):
		ind1 = np.random.randint(0,N)
		ind2 = np.random.randint(0,N)
		distance = (centers[ind1][0]-centers[ind2][0])**2 + (centers[ind1][1]-centers[ind2][1])**2
		distance = distance**0.5
		dist.append(distance)
	norm = entropy = float(sum(dist))/float(len(dist))
	entropy = np.std(dist)/norm
	return dist, entropy

def profile(arr, depth, j):
	'''
	profile as in helper.py in checkin2
	'''
	# arr is a depth sorted array
	# depth/width is the profile direction
	# points_d is the number of points across depth
	profile = []
	step = 3.
	current = 0.
	window = 10.
	depths = []
	while current+window<depth:
		a = search(arr, j, current, current+window)
		profile.append(len(a))
		current += step
		depths.append(current+window/2.)
		
	return depths, profile

def interpol_list(list1):
	'''
	simple interpolate for values in list. Never used. 
	'''
	nl = []
	for i in range(len(list1)-1):
		val = (list1[i]+list1[i+1])/2.
		nl.append(val)
	return nl


