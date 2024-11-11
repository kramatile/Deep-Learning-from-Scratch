import numpy as np
from Layers import Layer

class MaxPooling(Layer):
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        if stride is None: 
            self.stride = pool_size
        else: 
            self.stride = stride
        
    def forward(self, input):
        self.input = input
        self.row_step, self.column_step = self.stride
        self.pool_height, self.pool_width = self.pool_size

        output_height = (input.shape[1] - self.pool_height) // self.row_step + 1
        output_width = (input.shape[2] - self.pool_width) // self.column_step + 1
        output = np.zeros((input.shape[0],output_height, output_width))

        for k in range(input.shape[0]):  
            for i in range(0, output_height):
                for j in range(0, output_width):
                    start_i = i * self.row_step
                    start_j = j * self.column_step
                    element = input[k,start_i:start_i + self.pool_height, start_j:start_j + self.pool_width]
                    output[k, i, j] = np.max(element)                  
        
        return output

    def backward(self, output_gradient, learning_rate):
        output_grad = np.zeros_like(self.input)
        for k in range(output_gradient.shape[0]):
            for i in range(output_gradient.shape[1]):
                for j in range(output_gradient.shape[2]):
                    start_i = i * self.row_step
                    start_j = j * self.column_step
                    element = self.input[k,start_i:start_i + self.pool_height, start_j:start_j + self.pool_width]
                    ind = np.unravel_index(np.argmax(element, axis=None), element.shape)
                    output_grad[k,start_i + ind[0], start_j + ind[1]] += output_gradient[k,i,j]

        return output_grad