import numpy as np
import matplotlib.pyplot as plt
import random

random_numbers = np.loadtxt('homework2problem1parta_randomnumbers.txt')

x1_start, x2_start = 15315, 47836
mu, sigma = 0.0, 1.0

Ng = 64
N_max = Ng**2

def boxmuller(x1, x2, mu, sigma):
	
	#x1 and x2 are two sets of randomly generated numbers between 0 and 1
	N = len(x1)

	#gives distribution appropriate mean and standard deviation
	z1 = [ np.sqrt(-2*np.log(x1[i]))*np.cos(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]
	z2 = [ np.sqrt(-2*np.log(x1[i]))*np.sin(2*np.pi*x2[i]) * sigma + mu for i in range(N) ]

	return z1, z2

x1_parent = random_numbers[x1_start:x1_start+N_max]
x2_parent = random_numbers[x2_start:x2_start+N_max]

z1_parent, z2_parent = boxmuller(x1_parent, x2_parent, mu, sigma)

#Simpson's integration rule, \int_{a}^{b} f(x) dx with N sample points, also in 1b
def simpson(f, x_init, x_final, N_simp): 

    h = (x_final-x_init)/N_simp

    I = f(x_init) + f(x_final)

    odds = [4*f(x_init + k*h) for k in range(1,N_simp,2)]
    evens = [2*f(x_init + k*h) for k in range(2,N_simp,2)]
    I += sum(odds) + sum(evens)

    I *= h/3.
    
    return I

OM, OL = 0.3, 0.7
H0 = 67 #km/s/Mpc 


def H(z):
	
	return H0 * np.sqrt(OM*(1+z)**3 + OL)

def D(z, N): #need a number of sample points to get convergence of integration scheme as requested

	def f(zprime):

		return (1+zprime)/(H(zprime)**3)

	def Integrand(x): 

		'''
		x = (zprime - z)/(1+(zprime-z)), transformation from Mark Newman's Computatational Physics
		implicitly a variable of z but the for the integration routine the Integrand should only have one argument
		'''	

		return (1/(1-x)**2) * f((x/(1-x)) + z)

	eps_integration = 1e-14 #undefined at integal bound
	Integral = simpson(Integrand, 0, 1-eps_integration, N) #integrate Integrand from x = 0 to x = 1-eps

	return (5/2.) * OM * H(z) * H0**2 * Integral

def H_a(a):
		
	return H0 * np.sqrt(OM/(a**3) + OL)

def Ddot(a, N): #need a number of sample points to get convergence of integration scheme as requested

	def f(aprime):

		return 1/(a**3 * H_a(aprime)**3)

	eps_integration = 1e-14 #undefined at integal bounds
	Integral = simpson(f, eps_integration, a, N) #integrate Integrand from x = 0 to x = 1

	return 5*OM/2. * H0**2 * ( 1/(a**3 * H_a(a)**2) - 3*OM/(2*a**4) * (H0**2 / H_a(a)) * Integral )

def P(k):

	return k**(-2.)

def cofactor(k):

	return np.sqrt(P(k))/(k**2.)

#from 2

#k_x,y,z = (2*pi)/(Ng) * l, m, n from 0 to Ng/2

'''
having k = 0 corresponds to a wavlength -> infinity....
'''

kvals_pos = list(np.linspace(0.01, np.pi, int(Ng/2.)))
kvals_neg = list(np.linspace(-np.pi, -0.01, int(Ng/2.)))
kvals_tot = kvals_neg + kvals_pos

KX, KY = np.meshgrid(kvals_tot, kvals_tot)
	
kmags = [ np.sqrt(kvals_pos[j]**2 + kvals_tot[k]**2) for j in range(len(kvals_pos)) for k in range(len(kvals_tot)) ]
kmags = np.reshape(kmags, (int(Ng/2), Ng))


def Sq_components():

	#ak, bk = sqrt(P(k))/k**2 * Gauss(0,1)
	ck_righthalf = [ 0.5*( cofactor(kmags[j,k])*random.sample(z1_parent, 1)[0] - 1j*(cofactor(kmags[j,k]) * random.sample(z2_parent, 1)[0]) ) for j in range(len(kvals_pos)) for k in range(len(kvals_tot)) ]

	ck_righthalf = np.reshape(ck_righthalf, (int(Ng/2), Ng))

	#complex conjugate symmetry condition	
	ck_lefthalf = np.flip(np.flip(ck_righthalf, 1), 0)
	ck_lefthalf = np.conj(ck_lefthalf)
	ck_vals = np.concatenate((ck_lefthalf, ck_righthalf), axis=0)	

	Sq_x, Sq_y = np.fft.ifft2(ck_vals*KX), np.fft.ifft2(ck_vals*KY)

	#same issue as problem 2, casting as only the absolute values
	return abs(Sq_x), abs(Sq_y)

Sq_x, Sq_y = Sq_components()
J, K = Sq_x.shape

qvals_init = np.linspace(-int(Ng/2.)+0.5, int(Ng/2.)-0.5, Ng)

class particle():
	def __init__(self, position, momentum, q_init, Sq_init):
		#mass, x, and y for each particle
		self.position = position
		self.momentum = momentum
		self.q_init = q_init
		self.Sq_init = Sq_init

particle_positions = [ np.array( [q_x, q_y] ) for q_x in qvals_init for q_y in qvals_init ]
particle_Sqs = [ np.array( [Sq_x[j,k], Sq_y[j,k]] ) for j in range(J) for k in range(K) ]

particles = [ particle( np.array( [0., 0.] ), np.array( [0., 0.] ), np.array( [0., 0.] ), np.array( [0., 0.] ) ) for i in range(Ng**2) ]   

for i in range(len(particles)):
	particle = particles[i]
	particle.position = particle_positions[i]
	particle.q_init = particle_positions[i]
	particle.Sq_init = particle_Sqs[i]

#30 fps and lasts 3 seconds, so 90 snapshots

a_init, a_final, snapshots = 0.0025, 1.0, 90
delta_a = (a_final-a_init)/snapshots

scale_factor_vals = np.linspace(a_init, a_final, snapshots)

count = 0

first_ten_ys = [[] for i in range(10)]
first_ten_pys = [[] for i in range(10)]

for a in scale_factor_vals:

	xs, ys = [], []

	#N = 1024 from 4a, 4b
	cofactor_position = D((1/a - 1.), 1024)
	cofactor_momentum = -(a-delta_a/2.)**2 * Ddot(a-delta_a/2., 1024)

	for i in range(len(particles)):

		particle = particles[i]
		
		x, y = particle.position[:]
		px, py =  particle.momentum[:]
		q_vec = particle.q_init
		Sq_vec = particle.Sq_init

		xs.append(x)
		ys.append(y)

		if i < 10:
			
			fty = first_ten_ys[i]
			ftpy = first_ten_pys[i]

			fty.append(y)
			ftpy.append(py)

		j, k = int(x), int(y)

		particle.momentum = cofactor_momentum * Sq_vec
		particle.position = np.add( q_vec, cofactor_position * Sq_vec )

		x_new = particle.position[0]
		y_new = particle.position[1]

		#periodic boundary conditions
		if x_new < -Ng/2.:

			particle.position = np.array( [x_new + Ng, y_new] ) 

		if x_new > Ng/2.:

			particle.position = np.array( [x_new - Ng, y_new] ) 

		if y_new < -Ng/2.:

			particle.position = np.array( [x_new, y_new + Ng] ) 

		if y_new > Ng/2.:

			particle.position = np.array( [x_new, y_new - Ng] ) 

	plt.figure()
	plt.xlabel(r'$x$ (Mpc)', fontsize=12)
	plt.ylabel(r'$y$ (Mpc)', fontsize=12)
	plt.xlim(-int(Ng/2.), int(Ng/2.))
	plt.ylim(-int(Ng/2.), int(Ng/2.))
	plt.scatter(xs, ys, s=0.1, color='k')

	sc = str(count)

	plt.savefig('frame_%s.png'%(sc.zfill(4)))
	plt.close()		

	count += 1

plt.figure()

for i in range(10):

	fty = first_ten_ys[i]
	plt.plot(scale_factor_vals, fty, label='particle %i'%i)

plt.legend(loc='best')
plt.xlabel(r'$a$', fontsize=12)
plt.ylabel(r'$y$', fontsize=12)
plt.savefig('homework2problem4partcfigure1.pdf')
plt.close()

plt.figure()

for i in range(10):

	ftpy = first_ten_pys[i]
	plt.plot(scale_factor_vals, ftpy, label='particle %i'%i)

plt.legend(loc='best')
plt.xlabel(r'$a$', fontsize=12)
plt.ylabel(r'$p_{y}$', fontsize=12)
plt.savefig('homework2problem4partcfigure2.pdf')
plt.close()

print('Done with using Zeldovich approximation for 2D simulation')
