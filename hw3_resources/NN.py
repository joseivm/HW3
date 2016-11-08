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


	def predict(self,x):
		output = self.feedforward(x)
		return np.argmax(output)

	def predictX(self, X):
		n,d = X.shape
		Y = np.zeros(n)
		for i in xrange(n):
			output = self.feedforward(X[i])
			Y[i] = np.argmax(output)
		return Y



	def train(self, X, Y, l, max_iter = 10000):
		n,d = X.shape

		for c in xrange(int(max_iter/n)+1):
			indexes = np.arange(n)
			np.random.shuffle(indexes)
			errors = 0
			for i in indexes:
				x = X[i]
				y = Y[i]

				output = self.feedforward(x)
				y_pred = np.argmax(output)
				if y != y_pred:
					errors += 1

				self.compute_gradients(x,y)
				for layer in self.layers:
					layer.update_weights(l)
			print errors

	def compute_gradients(self, x, y):
		self.backprob(x,y)
		a = x
		for layer in self.layers:
			layer.compute_gradients(a)
			a = layer.a

	def feedforward(self,x):
		output = x
		for layer in self.layers:
			output = layer.feedforward(output)
		return softmax(output)

	def backprob(self, x, y):
		output = self.feedforward(x)
		d = output.shape[0]
		target = np.zeros(d)
		target[int(y)] = 1.0

		l = output - target
		layersReversed = self.layers[:]
		layersReversed.reverse()
		w = layersReversed[0].weights
		for layer in layersReversed[1:]:
			l = layer.backprob(w,l)
			w = layer.weights

def softmax(x):
	e_x = np.exp(x-np.max(x))
	return e_x/e_x.sum()


class NNLayer(object):

	def __init__(self, input_size, num_of_units):
		mu, sigma = 0, 1.0/np.sqrt(input_size)
		self.weights = np.random.normal(mu, sigma, (input_size, num_of_units))
		self.bias = np.zeros((1, num_of_units))[0]

		self.z = np.zeros(num_of_units)
		self.a = np.zeros(num_of_units)
		self.l = np.zeros(num_of_units)

		self.g_w = np.zeros(num_of_units)
		self.g_b = np.zeros(num_of_units)

	def update_weights(self, learning_rate):
		self.weights -= learning_rate*self.g_w
		self.bias -= learning_rate*self.g_b

	def feedforward(self, x):
		self.z = np.dot(self.weights.T, x) + self.bias
		self.a = ReLU(self.z)
		return self.a

	def backprob(self, w_next, l_next):
		diag = np.diagflat(np.sign(self.a))
		self.l = np.dot(np.dot(diag, w_next), l_next)
		return self.l

	def compute_gradients(self, a_prev):
		a = a_prev.copy()
		l = self.l.copy()
		a.shape = (len(a), 1)
		l.shape = (len(l), 1)

		self.g_w = np.dot(a, l.T)
		self.g_b = self.l

def ReLU(x):
	zero = np.zeros(len(x))
	return np.maximum(zero,x)

def compare(Y, Y_pred):
	n = len(Y)
	errors = 0
	for i in xrange(n):
		if Y[i][0] != Y_pred[i]:
			errors += 1
			# print Y[i][0], Y_pred[i]
	print 1-(errors*1.0)/n

if __name__ == '__main__':
	train = np.loadtxt('data/data_3class.csv')
	X = train[:, 0:2].copy()
	Y = train[:, 2:3].copy()
	# print X
	# print Y
	n,d = X.shape
	layers = [d, 10, 3]
	NN = Neural_Network(layers)
	NN.train(X,Y, l=1, max_iter=10000)

	Y_pred = NN.predictX(X)
	print Y_pred
	compare(Y, Y_pred)



