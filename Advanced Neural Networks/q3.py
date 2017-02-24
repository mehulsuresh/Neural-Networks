#IMPORTS
import math
import numpy as np
import random
from scipy.special import expit
act=0

#seed the random number generator to get same random number sequence
np.random.seed(42);
# inline float relu(float f) {
# 17	    return max(0.f, f);
# 18	}
# 19	
# 20	inline float relu_derivative(float f) {
# 21	    return f < 0 ? 0 : 1;
# 22	}

#activation functions and it's derivative
def activate(x):
	global act
	if act == 1:
		return 1.0/(1.0+np.exp(-x))#sigmoid
		pass
	elif act == 2:
		return math.tanh(x)
		pass
	elif act == 3:
		#return x * (x > 0.0)
		return 
		pass
	# elif act == 4:
	# 	if count == 0:
	# 		count=1;
	# 		return 1.0/(1.0+np.exp(-x))
	# 		pass
	# 	e = exp(x)
	# 	count = 0;
	# 	return (e / sum(e))
	# 	pass
	# elif act == 5:
	# 	if count == 0:
	# 		count=1;
	# 		return 1.0/(1.0+np.exp(-x))
	# 		pass
	# 	count = 0;
	# 	return np.log(1.0 + np.exp(x))
	# 	pass
	
	pass
def derivative(x):
	global act
	if act == 1:
		return x * (1.0 - x)#sigmoid
		pass
	elif act == 2:
		return 1 - x*x
		pass
	elif act == 3:
		#return x > 0.0
		return np.where(x > 0, 1.0, 0.0)
		pass
	# elif act == 4:
	# 	if count == 0:
	# 		count=1;
	# 		return x * (1.0 - x)
	# 		pass
	# 	count = 0;
	# 	m = softmax(x)
	# 	return (m - m^2)
	# 	pass
	# elif act == 5:
	# 	if count == 0:
	# 		count=1;
	# 		return expit(x)
	# 		pass
	# 	count = 0;
	# 	return x * (1.0 - x)
	# 	pass
	
	pass

# def relu(x):
#     return x * (x > 0.0)

# def relu_deriv(x):
#     return x > 0.0

# def tanh(x):
# #     return math.tanh(x)
# def softmax(x):
# 	vec = np.exp(x)
# 	return vec / np.sum(vec)

# def softmaxDerivative(x):
#  	return x*(1.0-x)

# # derivative for tanh sigmoid
# def dtanh(y):
#     return 1 - y*y
# #we use softplus because our output is outside the bounds of the sigmoid function [0,1
# #softplus goes from 0 to infinity
def softmax_activation(x):
	#return math.tanh(x)
	return np.log(1.0 + np.exp(x))
def softmax_derivative(x):
	#return 1 - x*x
    return expit(x)


#returns an empty matrix of size a x b
def weightArrays(a,b):
	mat = []
	for i in range(a):
		mat.append([0.0]*b)
	return mat
	pass

#returns a random number between a and b	
def rand(a,b):
	return (b-a)*np.random.random() + a
	pass

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self, num_inputs, num_hidden, num_outputs):
		super(NeuralNetwork, self).__init__()
		#number of input, output and hidden neurons
		self.num_inputs = num_inputs + 1 #+1 for bias mode
		self.num_outputs = num_outputs
		self.num_hidden = num_hidden
		self.learning_rate = 0.15
		self.momentum = 0.5
		self.rate_decay = 0.01
		# values for activation variables x,h,o etc for input, output and hidden neurons
		self.ai = [1.0]*self.num_inputs
		self.ah = [1.0]*self.num_hidden
		self.ao = [1.0]*self.num_outputs
        #create random weights for layer 1 and 2
		self.w1 = weightArrays(self.num_inputs, self.num_hidden)#weight values for layer 1
		self.w2 = weightArrays(self.num_hidden, self.num_outputs)#weight values for layer 2
        # set them to random vaules
		for i in range(self.num_inputs):
			for j in range(self.num_hidden):
				self.w1[i][j] = rand(-1, 1)
				pass
			pass
		for j in range(self.num_hidden):
			for k in range(self.num_outputs):
				self.w2[j][k] = rand(-2, 2)
				pass
			pass
		self.ci = np.zeros((self.num_inputs, self.num_hidden))
		self.co = np.zeros((self.num_hidden, self.num_outputs))	
	def forwardPropagate(self, patterns):
		count = 0;
		for p in patterns:
			count = count + 1;
			if count >= len(patterns)*(90/100): #Only test the network or the last 10% of the data
				print(p[0], '=>', self.calculate(p[0]),'-->',p[1])
				pass
			pass
		pass
	def calculate(self, inputs):
		if len(inputs) != self.num_inputs-1:
			raise ValueError('Wrong num_inputs!!')
		for i in range(self.num_inputs-1):
			self.ai[i] = inputs[i]#setting the value x1,x2 etc for the first layer calculations
			pass
        #FIRST LAYER
		for j in range(self.num_hidden):
			sum = 0.0
			for i in range(self.num_inputs):#finding w1x1 + w2x2 + . . .
				sum = sum + self.ai[i] * self.w1[i][j]
				pass
			self.ah[j] = activate(sum)#use sigmoid function to get output of first layer
		pass
    	#SECOND LAYER
		for k in range(self.num_outputs):
			sum = 0.0
			for j in range(self.num_hidden):
				sum = sum + self.ah[j] * self.w2[j][k]
			self.ao[k] = softmax_activation(sum)
			pass
		return self.ao[:]
	def trainNetwork(self, patterns):
		for x in range(100):
			count = 0
			mini_batch_size = 50
			n = len(patterns)
			error = 0
			#STOCHASTIC GRADIENT DESCENT
			random.shuffle(patterns)
			mini_batches = [patterns[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				for p in mini_batch:
					#print(p[0],'\n')
					self.calculate(p[0])
					error += self.backPropagation(p[1])
					pass
				self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
				#print('error %-.4f' % (error))
			pass
			print('error %-.4f' % (error))
			
		pass

	# def trainNetwork(self, patterns):
	# 	count = 0
	# 	error = 0
	# 	random.shuffle(patterns)
	# 	for p in patterns:
	# 		count = count + 1;
	# 		if count < len(patterns)*(90/100): #Only train the network or the first 90% of the data
	# 			self.calculate(p[0])
	# 			error = self.backPropagation(p[1])
	# 			print('error %-.4f' % error)
	# 			pass
	# 		pass
	# 	#self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
		
	# 	pass


	def backPropagation(self, targets):
		if len(targets) != self.num_outputs:
			raise ValueError('Wrong num_outputs!!')
			pass
		#FIND AND UPDATE WEIGHTS FOR LAYER 2
		layer2_partial_derivatives = [0.0] * self.num_outputs#create empty array to hold partial derivatives of layer 2
		for i in range(self.num_outputs):
			error = -(targets[i] - self.ao[i])#step1
			layer2_partial_derivatives[i] = softmax_derivative(self.ao[i]) * error#step2
		
		#UPDATE WEIGHTS FOR LAYER 1
		layer1_partial_derivatives = [0.0] * self.num_hidden#create empty array to hold partial derivatives of layer 1
		for j in range(self.num_hidden):
			error = 0.0
			for k in range(self.num_outputs):
				error += layer2_partial_derivatives[k] * self.w2[j][k]#step1
			layer1_partial_derivatives[j] = derivative(self.ah[j]) * error#step2
		#CHANGE VALUES FOR WEIGHT MATRICES w1 and w2
		for j in range(self.num_hidden):
			for k in range(self.num_outputs):
				change = layer2_partial_derivatives[k] * self.ah[j]
				self.w2[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum#layer-2 step-3
				self.co[j][k] = change
		for i in range(self.num_inputs):
			for j in range(self.num_hidden):
				change = layer1_partial_derivatives[j] * self.ai[i]
				self.w1[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum#layer-1 step-3
				self.ci[i][j] = change
		#CALCULATE ERROR WITH LOSS FUNCTION
		error = 0
		for i in range(self.num_outputs):
			y=targets[i]
			a=self.ao[i]
			#CROSS ENTROPY
			error += np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
			pass
		return error
		pass

def demo():
	global act
	#READING FILE AND FORMATTING DATA TO SUIT REQUIREMENTS
	f = open("data4.txt", "r")
	m = []
	pat = []
	count = 0;
	for line in f.readlines():
		m.append(line.split(" "));
	for x in m:
		pat.append([[float(x[0]),float(x[1]),float(x[2])],[float(x[3])]])
		#pat.append([[float(x[0]),float(x[1])],[float(x[2])]])
		pass
    #CREATING A NEURAL NETWORK WITH 3 INPUT NEURONS, 3 HIDDEN NEURONS AND 1 OUTPUT NEURON
	x = int(input("Enter the number of neurons for the input layer : "))
	y = int(input("Enter the number of neurons for the hidden layer : "))
	z = int(input("Enter the number of neurons for the output layer : "))
	act = 1;
	network = NeuralNetwork(x, y, z)
    # TRAINING NETWORK WITH DATA
	network.trainNetwork(pat)
    # TESING THE NETWORK AFTER TRAINING 
	network.forwardPropagate(pat)



if __name__ == '__main__':
	demo()