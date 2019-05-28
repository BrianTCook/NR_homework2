from __future__ import division
import numpy as np

#Simpson's integration rule, \int_{a}^{b} f(x) dx with N sample points, also in 1b
def simpson(f, x_init, x_final, N_simp): 

    h = (x_final-x_init)/N_simp

    I = f(x_init) + f(x_final)

    odds = [4*f(x_init + k*h) for k in range(1,N_simp,2)]
    evens = [2*f(x_init + k*h) for k in range(2,N_simp,2)]
    I += sum(odds) + sum(evens)

    I *= h/3.
    
    return I	

OM, OL = 0.3, 0.7 #matter and DE fractions
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

z = 50
N = 2

eps = 1. #just need to be greater than requested accuracy

while eps > 1e-5:

	Dnew, Dold = D(z, 2*N), D(z, N)
	eps = abs(Dnew-Dold)
	N *= 2

print('')
print('D(z = %i) is %.06f'%(z, Dnew))
print('')

with open('homework2problem4parta_result.txt', 'a') as f:
	f.write('D(z = %i) is %.04f \n'%(z, Dnew))
