import numpy as np

def MSE(yhat,ytrue):
    return np.mean((yhat-ytrue)**2)

def MSE_Derivative(yhat,ytrue):
    return (2/np.size(ytrue)) * (yhat - ytrue)