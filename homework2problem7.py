#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:04:11 2019

@author: BrianTCook
"""

from __future__ import division
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import h5py
filename = 'colliding.hdf5'
f = h5py.File(filename, 'r')

# Get the data
xyzs = list(f['PartType4']['Coordinates'])
ms = list(f['PartType4']['Masses'])

xyzs = [xyz.tolist() for xyz in xyzs]
ms = [m.tolist() for m in ms]

xs = [xyz[0] for xyz in xyzs]
ys = [xyz[1] for xyz in xyzs]

L = 150. #side length of box, most massive possible object
max_leaf, N_particles = 12, 1000

print_strs = []

class particle():
    def __init__(self, mass, x, y):
        #mass, x, and y for each particle
        self.mass = mass
        self.x = x
        self.y = y

particles_from_hdf5file = [ particle(ms[i], xs[i], ys[i]) for i in range(len(xyzs))]   
        
class Node():
    def __init__(self, xcen, ycen, width, height, particles):
        #position, size, particles, and children associated with a node
        self.xcen = xcen
        self.ycen = ycen
        self.width = height
        self.height = height
        self.particles = particles
        self.children = []
        
        '''
        calculates the n=0 multiple moment for each node
        regardless if it's a root, branch, or leaf
        '''
        
        self.zerothordermultiplemoment = np.sum( [particle.mass for particle in particles] )
    
def particles_in_node(xcenter, ycenter, width, height, particles):
    
    '''
    locates the particles within a node
    '''
    
    pin = [ particle for particle in particles if particle.x >= xcenter \
           and particle.x <= xcenter+width and particle.y >= ycenter and particle.y <= ycenter+height ]
    
    return pin
   
        
def subdivide_with_recursion(node, leaf_max, magic_particle):
    
    '''
    applies recursion to ensure a leaf node does 
    not have too many particles in it
    '''
    
    if len(node.particles) <= leaf_max:        
        return
    
    small_w = node.width/2.
    small_h = node.height/2.

    #northeast
    pin_ne = particles_in_node(node.xcen+small_w, node.ycen+small_h, small_w, small_h, node.particles)    
    node_ne = Node(node.xcen+small_w, node.ycen+small_h, small_w, small_h, pin_ne)
    
    if magic_particle in node_ne.particles:
        print_str = 'branch multiple moment is %.03f kg'%node_ne.zerothordermultiplemoment
        print(print_str)
        print_strs.append(print_str)
    
    subdivide_with_recursion(node_ne, leaf_max, magic_particle)
    
    #southeast
    pin_se = particles_in_node(node.xcen+small_w, node.ycen, small_w, small_h, node.particles)
    node_se = Node(node.xcen+small_w, node.ycen, small_w, small_h, pin_se)
    
    if magic_particle in node_se.particles:
        print_str = 'branch multiple moment is %.03f kg'%node_se.zerothordermultiplemoment
        print(print_str)
        print_strs.append(print_str)
    
    subdivide_with_recursion(node_se, leaf_max, magic_particle)
    
    #southwest
    pin_sw = particles_in_node(node.xcen, node.ycen, small_w, small_h, node.particles)
    node_sw = Node(node.xcen, node.ycen, small_w, small_h, pin_sw)
    
    if magic_particle in node_sw.particles:
        print_str = 'branch multiple moment is %.03f kg'%node_sw.zerothordermultiplemoment
        print(print_str)
        print_strs.append(print_str)
    
    subdivide_with_recursion(node_sw, leaf_max, magic_particle)
    
    #northwest
    pin_nw = particles_in_node(node.xcen, node.ycen+small_h, small_w, small_h, node.particles)    
    node_nw = Node(node.xcen, node.ycen+small_h, small_w, small_h, pin_nw)
    
    if magic_particle in node_nw.particles:
        print_str = 'branch multiple moment is %.03f kg'%node_nw.zerothordermultiplemoment
        print(print_str)
        print_strs.append(print_str)
    
    subdivide_with_recursion(node_nw, leaf_max, magic_particle)
    
    node.children = [node_ne, node_nw, node_se, node_sw]
        
def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children
    
class BH_quadtree():
	def __init__(self, leaf_max, N_particles):
		self.threshold = leaf_max
		self.particles = particles_from_hdf5file
		self.root = Node(0, 0, L, L, self.particles)

		magic_index = 100
		self.magic_particle = self.particles[magic_index]

		print('root multiple moment is %.03f kg'%(self.root.zerothordermultiplemoment))
		print_strs.append('root multiple moment is %.03f kg'%(self.root.zerothordermultiplemoment))


	def subdivide(self):
		subdivide_with_recursion(self.root, self.threshold, self.magic_particle)

	def plot(self):

		fig, ax = plt.subplots()
		kids = find_children(self.root)

		for kid in kids:
			ax.add_patch(Rectangle((kid.xcen, kid.ycen), kid.width, kid.height, linewidth=0.2, fill=False))

			xs = [ particle.x for particle in self.particles ]
			ys = [ particle.y for particle in self.particles ]

		plt.scatter(xs, ys, color='b', marker='*', s=1)
		plt.scatter(self.magic_particle.x, self.magic_particle.y, color='r', marker='o', s=12)
		plt.xlim(0,L)
		plt.ylim(0,L)
		plt.xlabel(r'$x$', fontsize=16)
		plt.ylabel(r'$y$', fontsize=16)
		plt.title('Barnes-Hut quadtree', fontsize=16)
		plt.gca().set_aspect('equal')
		plt.savefig('homework2problem7figure1.pdf')

original_tree = BH_quadtree(max_leaf, N_particles)
original_tree.subdivide()
original_tree.plot()

with open('homework2problem7_toprint.txt', 'w+') as f:
    for print_str in print_strs:
        f.write("%s \n"%print_str)
