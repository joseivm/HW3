'''
Neural Networks, Lets gooooooo


'''
import numpy as np 


class Neural_Network(object):


	def __init__(self, layer_sizes):
		self.layers = []
		input_size = layer_sizes[0]
		for layer_size in layer_sizes[1:]:
			self.layers.append(NNLayer(input_size, layer_size))
			input_size = layer_size

	# x should be an (n,1) array
	def feedforward(self,x):
		output = x
		for layer in self.layers:
			output = layer.feedforward(self, output)
		return softmax(output)


	def softmax(x):
		e_x = np.exp(x-np.max(x))
		return e_x/e_x.sum()

class NNLayer(object):

	def __init__(self, input_size, num_of_units):
		self.weights = np.random.rand(input_size, num_of_units)
		self.bias = np.random.rand(size,1)

	def feedforward(self, x):
		return ReLU(np.dot(self.weights.T, x) + self.bias)

	def ReLU(x):
		zero = np.zeros([len(x),1])
		return np.maximum(zero,x)

