import cupy as cp
#ReLU function
def ReLU(x):
    return  cp.maximum(0,x)

#sigmoid function
def sigmoid(x):
    y = 1/(1+cp.exp(-x))
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
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T

    x = x - cp.max(x) # 溢出对策
    return cp.exp(x) / cp.sum(cp.exp(x))