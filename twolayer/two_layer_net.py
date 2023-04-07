import sys, os
import numpy as np
from collections import OrderedDict
sys.path.append(os.pardir)
from common.function import *
from common.loss import *
from common.gradient import numerical_gradient
from common.layers import *

# build Two layer network
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        # Initialize weights
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size)
        self.params['B2'] = np.zeros(output_size)

        # Generate layers
        self.layers = OrderedDict()
        self.layers['affin_1'] = AffineLayer(self.params['W1'], self.params['B1'])
        self.layers['relu_1'] = ReluLayer()
        self.layers['affin_2'] = AffineLayer(self.params['W2'], self.params['B2'])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def gradiant(self, x, t):
        #forword
        self.loss(x, t)

        #backward
        dout = 1
        layers = list(self.layers).reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #setting
        grads = {}
        grads['W1'] = self.layers['affin_1'].dW
        grads['B1'] = self.layers['affin_1'].dB
        grads['W2'] = self.layers['affin_2'].dW
        grads['B2'] = self.layers['affin_2'].dB

        return  grads


