import numpy as np
import matplotlib.pyplot as plt
np.random.seed(121)

low, high, cellwidth, dim, N_particles = 0, 16, 1, 3, 1024

positions = np.random.uniform(low=low, high=high, size=(N_particles, dim))

gridpoint_locs = [[cellwidth*(i+1/2.), cellwidth*(j+1/2.), cellwidth*(k+1/2.)] for i in np.arange(low, high, cellwidth) \
			for j in np.arange(low, high, cellwidth) for k in np.arange(low, high, cellwidth)]

def separation_distance(vec1, vec2):
	'''
	both vectors are lists, must have same dimensionality
	'''
	return np.sqrt(np.sum([ (vec1[i] - vec2[i])**2 for i in range(len(vec1)) ]))

gridpoint_particles = [0.]*len(gridpoint_locs) #will have a count of the particles in each box
biggest_distance = np.sqrt( np.sum([(cellwidth / 2.)**2]*dim) ) #distance from center to corner of cell

for i in range(len(gridpoint_locs)):

	'''
	computes all of the particle distances for the ith cell, only keeps<
	the ones that are sufficiently close by
	'''

	gridpoint_loc = gridpoint_locs[i]

	separation_vecs = [ separation_distance(position, gridpoint_loc) for position in positions ]
	gridpoint_particles[i] = len( [ 1 for sep_vec in separation_vecs if sep_vec <= biggest_distance ] )

gridpoint_particles = np.reshape(gridpoint_particles, (int((high-low)/(cellwidth)), int((high-low)/(cellwidth)), int((high-low)/(cellwidth))))

zvals = [4, 9, 11, 14]

for i in range(len(zvals)):
	z = zvals[i]

	fig, ax = plt.subplots()
	im = ax.imshow(gridpoint_particles[:,:,z], extent=[low,high,low,high], vmin = 0, vmax = 8, origin='lower', aspect='equal')
	fig.colorbar(im)
	plt.title('z = %i slice'%(int(z)))
	plt.savefig('homework2problem5partafigure%i.pdf'%(i+1))
	plt.close()

print('Done allocating particles to cells')
