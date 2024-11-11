from Layers import Layer
import numpy as np
from scipy.signal import correlate2d, convolve2d


class Convolution2D(Layer):
    def __init__(self,input_size,kernel_size,filters):
        self.kernel_size = kernel_size
        self.filters = filters
        self.input_size = input_size
        self.input_channels = input_size[0]
        self.output_height = self.input_size[1] - self.kernel_size + 1
        self.output_width = self.input_size[2] - self.kernel_size + 1
        self.kernels = np.random.randn(filters,self.input_channels,kernel_size,kernel_size)
        self.biases = np.random.randn(filters,self.output_height,self.output_width)
    
    def forward(self, input):
        self.input = input
        Y = np.copy(self.biases)
        for i in range(self.filters):
            for j in range(self.input_channels):
                Y[i] += correlate2d(input[j],self.kernels[i,j],mode="valid")
        return Y
        
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        weights_gradient = np.zeros_like(self.kernels)
        for i in range(self.filters):
            for j in range(self.input_channels):
                weights_gradient[i][j] = correlate2d(self.input[j],output_gradient[i],mode="valid")
                input_gradient[j] += convolve2d(output_gradient[i],self.kernels[i][j],mode='full')

        self.biases -= learning_rate*output_gradient
        self.kernels -= learning_rate*weights_gradient
        return input_gradient