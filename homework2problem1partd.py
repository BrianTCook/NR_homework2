import numpy as np
import matplotlib.pyplot as plt
from astropy import stats
from scipy.special import erf

#from NR textbook
def QKP(lam):

        N = 100 #exp(-100) << machine precision
        summand = [(4 * j**2 * lam**2 - 1)*np.exp(-2 * j**2 * lam**2) for j in range(1,N)]

        return 2*np.sum(summand)

def siglevel(V, N):

	arg = (np.sqrt(N) + 0.155 + 0.24/np.sqrt(N)) * V

	return QKP(arg)

#from part B

def boxmuller(x1, x2, mu, sigma):
	
	#x1 and x2 are two sets of randomly generated numbers between 0 and 1
	N = len(x1)

	#gives distribution appropriate mean and standard deviation
	z1 = [ np.sqrt(-2*np.log(x1[i]))*np.cos(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]
	z2 = [ np.sqrt(-2*np.log(x1[i]))*np.sin(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]

	return z1, z2


'''
for arbitrary mu, sigma

def normaldistribution(mu, sigma, x):
		
	return np.sqrt(1/(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))

'''

#getting 2 sets of 1000 numbers as input for Box-Muller method
random_numbers = np.loadtxt('homework2problem1parta_randomnumbers.txt')

x1_start, x2_start = 15315, 47836
mu, sigma = 0.0, 1.0

N_max = int(1e5)

x1_parent = random_numbers[x1_start:x1_start+N_max]
x2_parent = random_numbers[x2_start:x2_start+N_max]

#executing Box-Muller transform
z1_parent, z2_parent = boxmuller(x1_parent, x2_parent, mu, sigma)

tests_myKuiper, tests_theirKuiper = [], []

'''
mu = 0 and sigma = 1 for this problem
'''

def CDF_theory(x):

	return 0.5*(1 + erf(x/np.sqrt(2)))

N_numbers = [int(10**i) for i in np.arange(1, 5, 0.1)] #have to round to nearest integer for some values of i

for N_numbers_val in N_numbers:

	z1, z2 = z1_parent[:N_numbers_val], z2_parent[:N_numbers_val]

	bins = np.linspace(mu-5*sigma, mu+5*sigma, int(N_numbers_val/2.)) #fixing the bin choice for the histogram of z1

	z1_histed_vals, z1_histed_bins = np.histogram(z1, bins=bins, density=True) #lists of length N and N+1
	z1_histed_cumulated = [np.sum(z1_histed_vals[:i]) for i in range(len(z1_histed_vals))]
	last_val = z1_histed_cumulated[len(z1_histed_cumulated)-1]

	z1_histed_cumulated = [ z1_h_c/last_val for z1_h_c in z1_histed_cumulated ]

	xs_for_theory = [0.5*(z1_histed_bins[i] + z1_histed_bins[i]) for i in range(len(z1_histed_bins)-1)]

	'''
	theoretical_distribution = [normaldistribution(mu, sigma, x) for x in xs_for_theory]
	theoretical_distribution_cumulated = [np.sum(theoretical_distribution[:i]) for i in range(len(theoretical_distribution))]
	
	there's a better way to do this, namely by using the error function
	'''

	theoretical_distribution_cumulated = [ CDF_theory(x) for x in xs_for_theory ]

	#for Kuiper, divides domain in two and finds D +/- for each half
	whole_range = range(len(z1_histed_cumulated))
	first_range = whole_range[:int(len(whole_range)/2.)]
	second_range = whole_range[int(len(whole_range)/2.):]

	Dminus = max([abs(z1_histed_cumulated[i] - theoretical_distribution_cumulated[i]) for i in first_range])
	Dplus = max([abs(z1_histed_cumulated[i] - theoretical_distribution_cumulated[i]) for i in second_range])
	V = Dminus + Dplus

	#same sort of thing as 1C
	tests_myKuiper.append(siglevel(V, N_numbers_val))
	tests_theirKuiper.append(stats.kuiper(z1, CDF_theory)[1])

#plotting
plt.figure()

plt.semilogx(N_numbers, tests_myKuiper, alpha=0.5, label='my Kuiper test')
plt.semilogx(N_numbers, tests_theirKuiper, alpha=0.5, label='their Kuiper test')

plt.xlabel('$N_{samples}$', fontsize=16)
plt.ylabel('Kuiper test result ($P$)', fontsize=16)
plt.legend(loc='best')
plt.savefig('homework2problem1partdfigure1.pdf')
plt.close()

print('Done with Kuiper test')
