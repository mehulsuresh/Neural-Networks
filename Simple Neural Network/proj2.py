#IMPORTS
import math
import numpy as np

#seed the random number generator to get same random number sequence
np.random.seed(42);

#activation function and it's derivative
def activate(x):
	return 1.0/(1.0+np.exp(-x))#sigmoid
	#return math.tanh(x);#tanH
	pass
def derivative(x):
	return x * (1.0 - x)
	#return 1.0 - x**2
	pass

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
		self.learning_rate = 0.5
		self.momentum = 0.5
		# values for activation variables x,h,o etc for input, output and hidden neurons
		self.ai = [1.0]*self.num_inputs
		self.ah = [1.0]*self.num_hidden
		self.ao = [1.0]*self.num_outputs
        #create random weights
		self.w1 = weightArrays(self.num_inputs, self.num_hidden)#weight values for layer 1
		self.w2 = weightArrays(self.num_hidden, self.num_outputs)#weight values for layer 2
        # set them to random vaules
		for i in range(self.num_inputs):
			for j in range(self.num_hidden):
				self.w1[i][j] = rand(-0.5, 0.5)
		for j in range(self.num_hidden):
			for k in range(self.num_outputs):
				self.w2[j][k] = rand(-0.5, 0.5)
				2
	def forwardPropagate(self, patterns):
		count = 0;
		for p in patterns:
			count = count + 1;
			if count >= len(patterns)*(90/100): #Only test the network or the last 10% of the data
				print(p[0], '->', self.calculate(p[0]))
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
			self.ao[k] = activate(sum)
			pass
		return self.ao[:]
	def trainNetwork(self, patterns):
		count = 0
		error = 0
		for p in patterns:
			count = count + 1;
			if count < len(patterns)*(90/100): #Only train the network or the first 90% of the data
				self.calculate(p[0])
				error = self.backPropagation(p[1])
				print('error %-.4f' % error)
				pass
			pass
		pass
	def backPropagation(self, targets):
		if len(targets) != self.num_outputs:
			raise ValueError('Wrong num_outputs!!')
			pass
		#FIND AND UPDATE WEIGHTS FOR LAYER 2
		layer2_partial_derivatives = [0.0] * self.num_outputs#create empty array to hold partial derivatives of layer 2
		for i in range(self.num_outputs):
			error = -(targets[i] - self.ao[i])#step1
			layer2_partial_derivatives[i] = derivative(self.ao[i]) * error#step2
		
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
				self.w2[j][k] -= (layer2_partial_derivatives[k] * self.ah[j])#layer-2 step-3
		for i in range(self.num_inputs):
			for j in range(self.num_hidden):
				self.w1[i][j] -= (layer1_partial_derivatives[j] * self.ai[i])#layer-1 step-3
		#CALCULATE ERROR WITH LOSS FUNCTION
		error = 0
		for i in range(self.num_outputs):
			error += 0.5 * ((targets[i] - self.ao[i]) ** 2)
			pass
		return error
		pass

def demo():
	#READING FILE AND FORMATTING DATA TO SUIT REQUIREMENTS
	f = open("data4.txt", "r")
	m = []
	pat = []
	count = 0;
	for line in f.readlines():
		m.append(line.split(" "));
	for x in m:
		pat.append([[float(x[0]),float(x[1]),float(x[2])],[float(x[3])]])
		pass
    #CREATING A NEURAL NETWORK WITH 3 INPUT NEURONS, 3 HIDDEN NEURONS AND 1 OUTPUT NEURON
	network = NeuralNetwork(3, 3, 1)
    # TRAINING NETWORK WITH DATA
	network.trainNetwork(pat)
    # TESING THE NETWORK AFTER TRAINING 
	network.forwardPropagate(pat)



if __name__ == '__main__':
	demo()