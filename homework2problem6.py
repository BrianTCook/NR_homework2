import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(3) #for initialization of weights vector

'''
getting input data
'''
data = np.loadtxt('GRBs.txt', unpack=True, usecols = (2,3,4,5,6,7,8))

redshift, T90, mass, sfr, metallicity, SSFR, AV = data

input_vecs = [ [redshift[i], T90[i], mass[i], metallicity[i], SSFR[i], AV[i]] for i in range(len(redshift)) ]
lengths = [ 1 if t >= 10. else 0 for t in T90] 

inputs, outputs = input_vecs, lengths

def activation(input_vec, weight_vec): #uses 
    summation = np.dot(input_vec, weight_vec[1:]) + weight_vec[0]
    #step function
    if summation > 0:
        return 0 #correctly classified
    else:
        return 1 #incorrectly classified

class perceptron(object):
    
	def __init__(self, len_input, threshold, learning_rate):
		self.len_input = len_input
		self.threshold = threshold
		self.learning_rate = learning_rate
		#need a vector of weights for classifictation (0 or 1), initialize randomly
		self.weights = np.random.rand(len_input+1, 2)  

	def train(self, inputs, outputs):
		count = 0
		while count < self.threshold:
			for input_vec, output in zip(inputs, outputs):
				weight_vec = self.weights[:,output]
				#determine if weights need to be adjusted
				activation_w_x = activation(input_vec, weight_vec) 
				self.weights[1:, output] += [ learning_rate * activation_w_x * iv for iv in input_vec ]
				self.weights[0, output] += learning_rate * activation_w_x
				count += 1

		return self.weights

len_input, threshold, learning_rate = len(inputs[0]), 200, 0.01

#sets up weights matrix on which the MNIST data can be applied
W = perceptron(len_input, threshold, learning_rate)

#applies the training data to find the weights
W_trained = W.train(inputs, outputs)

def get_output(inputs):
    outputs_from_W = []
    for x in inputs:
        w_dot_x = np.zeros(2) #finding the maximum value of \vec{w}\cdot\vec{x}
        for i in range(2): #iterating through short or long
            weight_vec = W_trained[:,i]
            w_dot_x[i] = (np.dot(x, weight_vec[1:]) + weight_vec[0])
        classification = np.argmax(w_dot_x)
        outputs_from_W.append(classification)
    return outputs_from_W


outputs_from_W = get_output(inputs)

plt.hist(outputs_from_W, alpha=0.5, label='Predicted')
plt.hist(lengths, alpha=0.5, label='Actual')
plt.ylabel(r'$N_{GRBs}$', fontsize=14)
plt.gca().set_xticks([0.05,0.95])
plt.gca().set_xticklabels(['short','long'])
plt.xlim(-1,2)
plt.legend(loc='best')
plt.savefig('homework2problem6figure1.pdf')
