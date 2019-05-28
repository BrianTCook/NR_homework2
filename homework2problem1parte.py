import numpy as np
import matplotlib.pyplot as plt

#from lecture
def Pks(z):

	if z < 1.18:

		a = np.exp(-np.pi**2 / (8*z**2))

		return np.sqrt(2*np.pi)/z * ( a + a**9 + a**25 )

	else:

		a = np.exp(-2*z**2)

		return 1 - 2*(a + a**4 + a**9)


#numerical recipes textbook
def siglevel(D, N):

	arg = (np.sqrt(N) + 0.12 + 0.11/np.sqrt(N)) * D

	return Pks(arg)

#from part B

def boxmuller(x1, x2, mu, sigma):
	
	#x1 and x2 are two sets of randomly generated numbers between 0 and 1
	N = len(x1)

	#gives distribution appropriate mean and standard deviation
	z1 = [ np.sqrt(-2*np.log(x1[i]))*np.cos(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]
	z2 = [ np.sqrt(-2*np.log(x1[i]))*np.sin(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]

	return z1, z2

Folkerts_random_numbers = np.asarray(np.loadtxt('randomnumbers.txt'))

#for comparison to given random number lists
random_numbers = np.loadtxt('homework2problem1parta_randomnumbers.txt')

x1_start, x2_start = 15315, 47836
mu, sigma = 0.0, 1.0

N_max = int(1e5)

x1_parent = random_numbers[x1_start:x1_start+N_max]
x2_parent = random_numbers[x2_start:x2_start+N_max]

#executing Box-Muller transform
z1_parent, z2_parent = boxmuller(x1_parent, x2_parent, mu, sigma)

xvals, yvals1, yvals2 = [], [], []

for i in range(len(Folkerts_random_numbers[0,:])):

	list_of_nums = Folkerts_random_numbers[:,i]

	L = 10000 #had issues with siglevel function for L = 1e5
	z1, z2 = z1_parent[:L], z2_parent[:L]
	list_of_nums = list_of_nums[:L]

	bins = np.linspace(mu-5*sigma, mu+5*sigma, int(L/2.)) #fixing the bin choice for the histogram of z1

	z1_histed_vals, z1_histed_bins = np.histogram(z1, bins=bins, density=True) #lists of length N and N+1
	f1_histed_vals, f1_histed_bins = np.histogram(list_of_nums, bins=bins, density=True) #lists of length N and N+1

	z1_histed_cumulated = [np.sum(z1_histed_vals[:i]) for i in range(len(z1_histed_vals))]
	f1_histed_cumulated = [np.sum(f1_histed_vals[:i]) for i in range(len(f1_histed_vals))]
	
	last_val_z1 = z1_histed_cumulated[len(z1_histed_cumulated)-1]
	last_val_f1 = f1_histed_cumulated[len(f1_histed_cumulated)-1]

	z1_histed_cumulated = [ z1_h_c/last_val_z1 for z1_h_c in z1_histed_cumulated ]
	f1_histed_cumulated = [ f1_h_c/last_val_f1 for f1_h_c in f1_histed_cumulated ]

	D = max([abs(z1_histed_cumulated[i] - f1_histed_cumulated[i]) for i in range(len(z1_histed_cumulated))])

	xvals.append(i)
	yvals1.append(D)
	yvals2.append(siglevel(D, L))

plt.figure()
plt.plot(xvals, yvals1)
plt.xlabel(r'$i$th list of random numbers', fontsize=12)
plt.ylabel(r'KS statistic (D)', fontsize=12)
plt.savefig('homework2problem1partefigure1.pdf')
plt.close()

plt.figure()
plt.plot(xvals, yvals2)
plt.xlabel(r'$i$th list of random numbers', fontsize=12)
plt.ylabel(r'KS significance result', fontsize = 12)
plt.savefig('homework2problem1partefigure2.pdf')
plt.close()
