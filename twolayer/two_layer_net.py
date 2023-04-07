import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.append(os.pardir)
from common.layers import *

# build Two layer network
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights, He_initialization
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)/np.sqrt(input_size/1.8)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size)/np.sqrt(hidden_size/2)
        self.params['b2'] = np.zeros(output_size)

        # Generate layers
        self.layers = OrderedDict()
        self.layers['affin_1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['relu_1'] = ReluLayer()
        self.layers['affin_2'] = AffineLayer(self.params['W2'], self.params['b2'])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        #forword
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #setting
        grads = {}
        grads['W1'] = self.layers['affin_1'].dW
        grads['b1'] = self.layers['affin_1'].db
        grads['W2'] = self.layers['affin_2'].dW
        grads['b2'] = self.layers['affin_2'].db

        return  grads


