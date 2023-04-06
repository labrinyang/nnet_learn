import numpy as np

#ReLU function
def ReLU(x):
    return  np.maximum(0,x)

#sigmoid function
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

#sigmoid_grad function
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

#step function
def identity_function(x):
    return x

#softmax function
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))