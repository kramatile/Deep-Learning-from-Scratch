from Activation import Activation
import numpy as np

class Sigmoid(Activation):
    def __init__(self):
        def activation(input):
            output = 1 / (1+np.exp(-input))
            return output
        
        def activation_prime(input):
            a = activation(input)
            derivative = (1 - a) * a
            return derivative
            
        super().__init__(activation,activation_prime)

class ReLU(Activation):
    def __init__(self):
        def activation(input):
            output = np.where(input > 0,input,0)
            return output
        
        def activation_prime(input):
            return np.where(input > 0,1,0)

        super().__init__(activation,activation_prime)

class Tanh(Activation):
    def __init__(self):
        def activation(input):
            output = (np.exp(input)-np.exp(-input))/(np.exp(input)+np.exp(-input))
            return output
        
        def activation_prime(input):
            derivative = 1 - (activation(input)**2)
            return derivative

        super().__init__(activation,activation_prime)
