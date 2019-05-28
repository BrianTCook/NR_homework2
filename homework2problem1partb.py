import numpy as np
import matplotlib.pyplot as plt

#transforms 2D uniform distribution to 2D bivariate normal distribution with mean/standard deviation mu, sigma
def boxmuller(x1, x2, mu, sigma):
	
	#x1 and x2 are two sets of randomly generated numbers between 0 and 1
	N = len(x1)

	#gives distribution appropriate mean and standard deviation
	z1 = [ np.sqrt(-2*np.log(x1[i]))*np.cos(2*np.pi*x2[i]) * sigma + mu for i in range(N) ] 
	z2 = [ np.sqrt(-2*np.log(x1[i]))*np.sin(2*np.pi*x2[i]) * sigma + mu for i in range(N) ] 

	return z1, z2

#getting 2 sets of 1000 numbers as input for Box-Muller method
random_numbers = np.loadtxt('homework2problem1parta_randomnumbers.txt')

x1_start, x2_start = 15315, 47836

x1 = [random_numbers[i] for i in range(x1_start, x1_start+1000)]
x2 = [random_numbers[i] for i in range(x2_start, x2_start+1000)]

mu, sigma = 3.0, 2.4

def normaldistribution(mu, sigma, x):
	
	return np.sqrt(1/(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))

#executing Box-Muller transform
z1, z2 = boxmuller(x1, x2, mu, sigma)

#getting theoretical normal distribution
xs = np.linspace(mu-5*sigma, mu+5*sigma, 1000)
ys = [normaldistribution(mu, sigma, x) for x in xs]

#plotting
plt.figure()

bins = np.linspace(mu-5*sigma, mu+5*sigma, 100) #fixing the bin choice for the histogram of z1
plt.hist(z1, bins=bins, color = 'g', linewidth=1, histtype='step', density=True, label=r'Box-Muller Numbers ($z_{1}$)')
plt.plot(xs, ys, 'r', linewidth=1, label='Normal Distribution')
plt.annotate(r'$\mu = %.01f$'%(mu), xy = (0.1, 0.2), xycoords = 'axes fraction', fontsize=12)
plt.annotate(r'$\sigma = %.01f$'%(sigma), xy = (0.1, 0.1), xycoords = 'axes fraction', fontsize=12)

#stdev benchmarks
for i in range(1,5):
	plt.axvline(x=mu+i*sigma, linewidth=1, linestyle='--', label=r'$\mu + %i\sigma$'%(i))

plt.legend(loc='best', fontsize = 8)
plt.xlim(mu - 5*sigma, mu + 5*sigma)
plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel(r'$n(x) \Delta x / \sum n(x) \Delta x$', fontsize = 16)
plt.tight_layout()
plt.savefig('homework2problem1partbfigure1.pdf')

print('Done comparing Box-Muller transformed random numbers to normal distribution')
