import numpy as np
import matplotlib.pyplot as plt
import random

#transforms 2D uniform distribution to 2D bivariate normal distribution with mean/standard deviation mu, sigma
def boxmuller(x1, x2, mu, sigma):
	
	#x1 and x2 are two sets of randomly generated numbers between 0 and 1
	N = len(x1)

	#gives distribution appropriate mean and standard deviation
	z1 = [ np.sqrt(-2*np.log(x1[i]))*np.cos(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]
	z2 = [ np.sqrt(-2*np.log(x1[i]))*np.sin(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]

	return z1, z2

random_numbers = np.loadtxt('homework2problem1parta_randomnumbers.txt')
random_numbers = list(random_numbers)

def sigma_func(k,n):
    return np.sqrt(k**n)

nvals = [-1, -2, -3]

N_k = 1024
xmin, xmax = 100./N_k , 100. #Mpc, box size of 100 Mpc
kmin, kmax = 2*np.pi/xmax, 2*np.pi/xmin

kvals_pos = list(np.linspace(kmin, kmax, int(N_k/2)))
kvals_neg = list(np.linspace(-kmax, -kmin, int(N_k/2)))
kvals_tot = kvals_neg + kvals_pos
	
kmags = [ np.sqrt(kvals_pos[j]**2 + kvals_tot[k]**2) for j in range(len(kvals_pos)) for k in range(len(kvals_tot)) ]
kmags = np.reshape(kmags, (int(N_k/2), N_k))

mu = 0.

for i in range(len(nvals)):

	n = nvals[i]
	    
	four_plane_righthalf = [ boxmuller([random.sample(random_numbers, 1)[0]], [random.sample(random_numbers, 1)[0]], mu, sigma_func(kmags[j,k],n))[0][0] \
			    + 1j * boxmuller([random.sample(random_numbers, 1)[0]], [random.sample(random_numbers, 1)[0]], mu, sigma_func(kmags[j,k],n))[1][0] \
			    for j in range(len(kvals_pos)) for k in range(len(kvals_tot)) ]


	four_plane_righthalf = np.reshape(four_plane_righthalf, (int(N_k/2), N_k))

	#complex conjugate symmetry condition	
	four_plane_lefthalf = np.flip(np.flip(four_plane_righthalf, 1), 0)
	four_plane_lefthalf = np.conj(four_plane_lefthalf)

	four_plane = np.concatenate((four_plane_lefthalf, four_plane_righthalf), axis=0)	
	four_plane_inverse = np.fft.ifft2(four_plane)

	plt.figure()
	im = plt.imshow(abs(four_plane_inverse), origin='lower',extent=[xmin, xmax, xmin, xmax])
	plt.xlabel(r'$x_{1}$ (Mpc)', fontsize=16)
	plt.ylabel(r'$x_{2}$ (Mpc)', fontsize=16)
	plt.title(r'$\sigma^{2} \propto k^{%i}$'%n, fontsize=16)
	plt.colorbar(im)
	plt.savefig('homework2problem2figure%i.pdf'%(i+1))
	plt.close()

print('Done with the Gaussian random fields')
