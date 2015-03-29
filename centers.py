import os
import numpy as np
import scipy.optimize as opt 
import scipy.spatial.distance as d
import matplotlib.pyplot as plt

'''
Fit to graph ignoring and considering edge effects
Also calculate goodness of fit and store in Results.txt
'''

def logistic(x, mu, s, A):
	return A*(1/4.0/s) * np.cosh((x-mu)/2.0/s)**(-2)

def boltzmann(x,a, A):
	return A*(2.0/np.pi)**0.5 * (x**2 * np.exp(-x**2/2.0/(a**2)))/a**3

def gumbel(x, mu,b, A):
	z = (x-mu)/b
	return A*np.exp(-1.0*(z + np.exp(-z)))/b

def hist_data(array, block):
	hist, bins = np.histogram(array, bins=block)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	return center, hist

def runner(pathb, pathg, pathl, file, counter):
	log_init = (53.7873757895, 20.3443103825, 6140956.95601)
	bltz_init = (37.5461720445, 5133049.50192)
	limit_x = limit_y = 140
	position = np.loadtxt(file, delimiter = ' ')
	x = position[:,0]
	y = position[:,1] 
	c = d.pdist(position)
	xvals, yvals = hist_data(c, 50) # Get histogram data

	# Cure Fit Logistic
	popt_log, pcov_log  = opt.curve_fit(logistic, xvals, yvals, p0 = log_init)

	y_log = logistic(xvals, popt_log[0], popt_log[1], popt_log[2])
	y_log_avg = np.sum(y_log)/float(len(y))

	SSreg_log = np.sum((y_log_avg - y_log)**2)
	SStot_log = np.sum((y_log_avg - yvals)**2)

	R2_log = SSreg_log/float(SStot_log)

	plt.plot(xvals, yvals)
	plt.plot(xvals, y_log)
	plt.title('Logistic Fit Graph - R^2 is {0}'.format(R2_log))
	plt.xlabel('$l$ (in units radius)')
	plt.ylabel('Frequency')
	plt.savefig(pathl + 'Logistic{0}.png'.format(counter))
	# plt.show()

	# Cure Fit Boltzman
	popt_b, pcov_b  = opt.curve_fit(boltzmann, xvals, yvals, p0= bltz_init)

	y_b = boltzmann(xvals, popt_b[0], popt_b[1])
	y_b_avg = np.sum(y_b)/float(len(y))

	SSreg_b = np.sum((y_b_avg - y_b)**2)
	SStot_b = np.sum((y_b_avg - yvals)**2)

	R2_b = SSreg_b/float(SStot_b)

	plt.close()
	plt.plot(xvals, yvals)
	plt.plot(xvals, y_log)
	plt.title('Boltzmann Fit Graph - R^2 is {0}'.format(R2_b))
	plt.xlabel('$l$ (in units radius)')
	plt.ylabel('Frequency')
	plt.savefig(pathb+'Boltzmann{0}.png'.format(counter))

	# Cure Fit Gumbel
	popt_g, pcov_g  = opt.curve_fit(gumbel, xvals, yvals , p0 = log_init)

	y_g = gumbel(xvals, popt_g[0], popt_g[1], popt_g[2])
	y_g_avg = np.sum(y_g)/float(len(y))

	SSreg_g = np.sum((y_g_avg - y_g)**2)
	SStot_g = np.sum((y_g_avg - yvals)**2)

	R2_g = SSreg_g/float(SStot_g)

	plt.close()
	plt.plot(xvals, yvals)
	plt.plot(xvals, y_g)
	plt.title('Gumbel Fit Graph - R^2 is {0}'.format(R2_g))
	plt.xlabel('$l$ (in units radius)')
	plt.ylabel('Frequency')
	plt.savefig(pathg + 'Gumbel{0}.png'.format(counter))
	plt.close()
	return R2_b, R2_log, R2_g

def runnermod(pathb, pathg, pathl, file, counter):
	log_init = (53.7873757895, 20.3443103825, 6140956.95601)
	bltz_init = (37.5461720445, 5133049.50192)
	limit_x = limit_y = 140
	position = np.loadtxt(file, delimiter = ' ')
	x = position[:,0]
	y = position[:,1] 
	c = d.pdist(position)
	xvals, yvals = hist_data(c, 50) # Get histogram data

	# Cure Fit Logistic
	popt_log, pcov_log  = opt.curve_fit(logistic, xvals[xvals<95], yvals[xvals<95], p0 = log_init)

	y_log = logistic(xvals, popt_log[0], popt_log[1], popt_log[2])
	y_log_avg = np.sum(y_log)/float(len(y))

	SSreg_log = np.sum((y_log_avg - y_log)**2)
	SStot_log = np.sum((y_log_avg - yvals)**2)

	R2_log = SSreg_log/float(SStot_log)

	plt.plot(xvals, yvals)
	plt.plot(xvals, y_log)
	plt.title('Logistic Fit Graph - R^2 is {0}'.format(R2_log))
	plt.xlabel('$l$ (in units radius)')
	plt.ylabel('Frequency')
	plt.savefig(pathl + 'Logistic_mod{0}.png'.format(counter))
	# plt.show()

	# Cure Fit Boltzman
	popt_b, pcov_b  = opt.curve_fit(boltzmann, xvals[xvals<95], yvals[xvals<95], p0= bltz_init)

	y_b = boltzmann(xvals, popt_b[0], popt_b[1])
	y_b_avg = np.sum(y_b)/float(len(y))

	SSreg_b = np.sum((y_b_avg - y_b)**2)
	SStot_b = np.sum((y_b_avg - yvals)**2)

	R2_b = SSreg_b/float(SStot_b)

	plt.close()
	plt.plot(xvals, yvals)
	plt.plot(xvals, y_log)
	plt.title('Boltzmann Fit Graph - R^2 is {0}'.format(R2_b))
	plt.xlabel('$l$ (in units radius)')
	plt.ylabel('Frequency')
	plt.savefig(pathb+'Boltzmann_mod{0}.png'.format(counter))

	# Cure Fit Gumbel
	popt_g, pcov_g  = opt.curve_fit(gumbel, xvals[xvals<95], yvals[xvals<95] , p0 = log_init)

	y_g = gumbel(xvals, popt_g[0], popt_g[1], popt_g[2])
	y_g_avg = np.sum(y_g)/float(len(y))

	SSreg_g = np.sum((y_g_avg - y_g)**2)
	SStot_g = np.sum((y_g_avg - yvals)**2)

	R2_g = SSreg_g/float(SStot_g)

	plt.close()
	plt.plot(xvals, yvals)
	plt.plot(xvals, y_g)
	plt.title('Gumbel Fit Graph - R^2 is {0}'.format(R2_g))
	plt.xlabel('$l$ (in units radius)')
	plt.ylabel('Frequency')
	plt.savefig(pathg + 'Gumbel_mod{0}.png'.format(counter))
	plt.close()
	return R2_b, R2_log, R2_g

a = os.listdir("C:\Users\Abhinav\Desktop\AdvPhyLab\checkin3\AdvPhyLab")
pathy = "C:\\Users\\Abhinav\\Desktop\\AdvPhyLab\\checkin3\\pure_data_analysis\\"
bpath = "C:\\Users\\Abhinav\\Desktop\\AdvPhyLab\\checkin3\\pure_data_analysis\\Boltzmann\\"
lpath = "C:\\Users\\Abhinav\\Desktop\\AdvPhyLab\\checkin3\\pure_data_analysis\\Logistic\\"
gpath = "C:\\Users\\Abhinav\\Desktop\\AdvPhyLab\\checkin3\\pure_data_analysis\\Gumbel\\"
filenames = []
for name in a:
	if '_centers' in name:
		filenames.append(name)
bolt = []
loggy = []
gummy = []

boltm = []
loggym = []
gummym = []

l = 0
for filename in filenames:
	R2_b, R2_log, R2_g = runner(bpath, gpath, lpath, pathy+ filename, l)
	R2_bm, R2_logm, R2_gm = runnermod(bpath, gpath, lpath, pathy+ filename, l)
	bolt.append(R2_b)
	loggy.append(R2_log)
	gummy.append(R2_g)
	boltm.append(R2_bm)
	loggym.append(R2_logm)
	gummym.append(R2_gm)
	l+=1

f = file("Results.txt", 'a')
np.savetxt(f, [bolt, loggy, gummy, boltm, loggym, gummym])
def avg(arr):
	return sum(arr)/float(len(arr))
print 'Boltz: ', avg(bolt),'Logist: ', avg(loggy),'Gumbel: ', avg(gummy)

print 'Boltz_mod: ', avg(boltm),'Logist_mod: ', avg(loggym),'Gumbel_mod: ', avg(gummym)



