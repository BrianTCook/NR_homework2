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

OM, OL = 0.3, 0.7
H0 = 67 #km/s/Mpc

def H(a):
		
	return H0 * np.sqrt(OM/(a**3) + OL)

def Ddot(a, N): #need a number of sample points to get convergence of integration scheme as requested

	def f(aprime):

		return 1/(aprime**3 * H(aprime)**3)

	eps_integration = 1e-14 #undefined at integal bounds
	Integral = simpson(f, eps_integration, 1/a - 1, N) #integrate Integrand from x = 0 to x = 1

	return 5*OM/2. * H0**2 * ( 1/(a**3 * H(a)**2) - 3*OM/(2*a**4) * (H0**2 / H(a)) * Integral )

z = 50
a = 1/(1+50)

N = 2

eps = 1. #just need to be greater than requested accuracy

while eps > 1e-5:

	Ddotnew, Ddotold = Ddot(a, 2*N), Ddot(a, N)
	eps = abs(Ddotnew-Ddotold)

	N *= 2

print('')
print('Ddot(a(z = 50)) (using H0 in inverse years) is', Ddotnew)
print('')

with open('homework2problem4partb_result.txt', 'a') as f:
	f.write("Ddot(a(z = 50)) (using H0 in inverse years) is %.04e \n"%(Ddotnew))
