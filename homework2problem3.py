from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

H0, W0 = 7.16e-11, 1. #yr^{-1}, unitless

def f(r,t):

	D = r[0]
	Ddot = r[1]

	'''
	can plug in values of a(t) and timederiv_a(t) to get expression independent of H0
	'''

	fD = Ddot
	fDdot = -(4/3.) * (1/t) * Ddot + (2*W0/3.) * (1/t**2) * D

	return np.array([fD, fDdot],float)

def rk4(f,r,t,h):
    
	k1 = h*f(r,t)
	k2 = h*f(r+0.5*k1,t+0.5*h)
	k3 = h*f(r+0.5*k2,t+0.5*h)
	k4 = h*f(r+k3,t+h)

	return (k1+2*k2+2*k3+k4)/6

def ODEsolver_usingRK4(init_conditions, a, b, N):
    
	h = (b-a)/N

	r = init_conditions
	tvals, Dvals = [], []

	for t in np.arange(a,b,h):
		tvals.append(t)
		Dvals.append(r[0])
		r += rk4(f,r,t,h)

	return tvals, Dvals

#initial conditions
case1_ics = np.array([3,2],float)
case2_ics = np.array([10,-10],float)
case3_ics = np.array([5,0],float)

case1_AB = np.array([3,0],float)
case2_AB = np.array([0,10],float)
case3_AB = np.array([3,2],float)

'''
Need analytic_D(D_init, Ddot_init, t) expression to compare for each of the three cases
'''

def D_analytic(t, A, B):

	return A * t**(2/3.) + B * t**(-1.)

cases = [[case1_ics, 'Case 1', case1_AB], [case2_ics, 'Case 2', case1_AB], [case3_ics, 'Case 3', case1_AB]]

t_init, t_final, s = 1., 1000., 100
N = int((t_final-t_init)*s) #samples ~s times a year

plt.figure()

count = 1

for case, case_str, case_AB in cases:

	A, B = case_AB

	tvals, Dvals = ODEsolver_usingRK4(case, 1., 1000., N)
	analytic_vals = [ D_analytic(t, A, B) for t in tvals ]

	plt.figure()

	plt.loglog(tvals, Dvals, alpha=0.5, label='RK4')
	plt.loglog(tvals, Dvals, alpha=0.5, linestyle='--', label='analytic')

	plt.title(case_str, fontsize=24)
	plt.legend(loc='best')
	plt.xlabel(r'$t$', fontsize=24)
	plt.ylabel(r'$D(t)$', fontsize=24)
	plt.tight_layout()
	plt.savefig('homework2problem3figure%i.pdf'%count)

	count += 1

print('Done solving the ODE')
        
