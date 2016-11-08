'''
Neural Networks, Lets gooooooo


'''
import numpy as np 


class Neural_Network(object):


	def __init__(self, layer_sizes):
		self.num_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.biases = [np.random.rand(size,1) for size in layer_sizes[1:]]
		self.weights = [np.random.rand(x,y) for x,y in zip(layer_sizes[:-1],layer_sizes[1:0])]


	# x should be an (n,1) array
	def feedforward(self,x):
		inputt = x
		for W,b in zip(self.weights,self.biases)[:-1]:
			inputt = ReLU(np.dot(W.T,inputt)+b)

		W = self.weights[-1]
		b = self.biases[-1]
		output softmax(np.dot(W.T,inputt)+b)

		return output

	def ReLU(x):
		zero = np.zeros([len(x),1])
		return np.maximum(zero,x)

	def softmax(x):
		return np.exp(x)/np.sum(np.exp(x))

