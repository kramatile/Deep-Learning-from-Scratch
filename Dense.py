import numpy as np
from Layers import Layer
from math import sqrt

class Dense(Layer):
    def __init__(self,input_size,output_size,initialization='default',seed=24):
        np.random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        # The size of the weight is : output_size (rows) X input_size (columns)
        initializations = {'xavier':self.XavierInitialization,'He':self.HeInitialization}
        self.weight = initializations.get(initialization,lambda: np.random.uniform(-1, 1, (output_size, input_size)))()
        # The size of the bias is : output_size (rows) x 1
        self.bias = np.random.uniform(-1,1,(output_size,1))
    
    # Forward  pass
    def forward(self, input):
        # Y = W*X + B
        self.input = input
        self.output = np.matmul(self.weight,self.input) + self.bias
        return self.output
    
    # Backward pass
    def backward(self, output_gradient, learning_rate):
        # Backpropagation Algorithm
        input_gradient = np.matmul(self.weight.T ,output_gradient)

        weight_gradient = np.matmul(output_gradient ,self.input.T) 
        #bias_gradient = output_gradient

        self.weight -= learning_rate*weight_gradient
        self.bias -= learning_rate*output_gradient
        return input_gradient

    def XavierInitialization(self):
        bound = sqrt(6)/(self.input_size + self.output_size)
        weight = np.random.uniform(-bound,bound,(self.output_size,self.input_size))
        return weight

    def HeInitialization(self):
        variance =sqrt(2/self.input_size)
        weight = np.random.normal(0,variance,(self.output_size,self.input_size))
        return weight
        