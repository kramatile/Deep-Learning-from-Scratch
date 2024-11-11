import numpy as np

def MSE(yhat,ytrue):
    return np.mean((yhat-ytrue)**2)

def MSE_Derivative(yhat,ytrue):
    return (2/np.size(ytrue)) * (yhat - ytrue)

def BinaryCrossEntropy(yhat,ytrue):
    return -np.mean(ytrue*np.log(yhat)+(1-ytrue)*np.log((1-yhat)))
    

def BinaryCrossEntropy_Derivative(yhat,ytrue):
    return (((1-ytrue)/(1-yhat))-(ytrue/yhat))/np.size(ytrue)