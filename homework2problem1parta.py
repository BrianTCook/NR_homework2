from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Problem 2, Part A 

def rng(seed):
	
	#must use unsigned integers for this RNG scheme (XOR shift + MWC) to work
	seed = np.uint64(seed)

	#good values for 64-bit XOR-shift method according to lecture
	a1, a2, a3 = np.uint64(21), np.uint64(35), np.uint64(4) 	

	#good values for MWC according to lecture
	a = np.uint64(4294957665) 
	mwc_b, mwc_c = np.uint64(2**32 - 1), np.uint(32)
	
	#MWC scheme provided in lecture with base b = 2**32
	seed = a * (seed & mwc_b) + (seed >> mwc_c)

	#XOR 64-bit shift method
	seed = seed ^ (seed << a1)
	seed = seed ^ (seed >> a2)
	seed = seed ^ (seed << a3)

	#one more MWC for good measure
	seed = a * (seed & mwc_b) + (seed >> mwc_c)

	return seed

seed = 91230495414757263 #needs to be an integer

print_strs = []

print_str = 'The seed for this list of random numbers is %i.'%seed
print(print_str)
print_strs.append(print_str)

with open('homework2problem1parta_print.txt', 'a') as f:
	for print_str in print_strs:
		f.write("%s \n" % print_str)

N_total = 1000000
rns = [0 for i in range(N_total)] #need to initialize rns list

rns[0] = seed

#must use for loop as list comprehension would not work in this case, e.g. f[i] = [g(f[i-1]) for i in range(1, N_total)]
for i in range(1,N_total):
	rns[i] = rng(rns[i-1])

#reducing the list of pseudo-random integers to floats 0 < {rns} <= 1
max_rns = max(rns)
random_numbers = [rns[i]/max_rns for i in range(N_total)]

#saving them for later
with open('homework2problem1parta_randomnumbers.txt', 'a') as f:
	for rn in random_numbers:
		f.write("%.06f \n" % rn)

#first plot
N_scatter = 1000
X_scatter = random_numbers[0:N_scatter-1]
Y_scatter = random_numbers[1:N_scatter]

cmap = cm.rainbow(np.linspace(0.0, 1.0, 1000))

plt.figure()
plt.scatter(X_scatter, Y_scatter, c=cmap, s=8)
plt.xlabel('$\{x_{i}\}$', fontsize=20)
plt.ylabel('$\{x_{i+1}\}$', fontsize=20)
plt.tight_layout()
plt.savefig('homework2problem1partafigure1.pdf')
plt.close()

#second plot

plt.figure()

Y_scatter_figuretwo = random_numbers[0:N_scatter]

plt.scatter(range(N_scatter), Y_scatter_figuretwo, c=cmap, s=8)
plt.xlabel('Index', fontsize=20)
plt.ylabel('$\{x_{i}\}$', fontsize=20)
plt.tight_layout()
plt.savefig('homework2problem1partafigure2.pdf')
plt.close()

#third plot
plt.figure()
plt.hist(random_numbers, bins = 20, histtype = 'step', label='Random Number Histogram')

#ensuring each bin doesn't deviate beyond 1 sigma
binned_random_numbers = np.histogram(random_numbers, bins=20)[0]
rns_mean, rns_stdev = np.mean(binned_random_numbers), np.std(binned_random_numbers)

plt.axhline(y=rns_mean + rns_stdev, color='g', linestyle='--', linewidth=1, label=r'$\mu + 1\sigma$')
plt.axhline(y=rns_mean - rns_stdev, color='g', linestyle='--', linewidth=1, label=r'$\mu - 1\sigma$')

plt.axhline(y=rns_mean + 2*rns_stdev, color='r', linestyle='--', linewidth=1, label=r'$\mu + 2\sigma$')
plt.axhline(y=rns_mean - 2*rns_stdev, color='r', linestyle='--', linewidth=1, label=r'$\mu - 2\sigma$')

plt.xlim(0,1)
plt.ylim(rns_mean - 5*rns_stdev, rns_mean + 5*rns_stdev)
plt.xlabel('random number value', fontsize=20)
plt.ylabel('N(generated numbers)', fontsize=20)
plt.legend(loc='best', fontsize = 8)
plt.savefig('homework2problem1partafigure3.pdf')
plt.close()
