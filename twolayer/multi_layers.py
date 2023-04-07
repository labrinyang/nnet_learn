import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.function import *

class MultiLayerNet:
    '''MultiLayerNet
    This class implements a multi-layer neural network with numerous hidden layers to boost its performance,
    while also introducing new layers to expand its capabilities. With this class, you have the flexibility to
    tailor the depth of the network and choose whether to incorporate extra layers, such as batch normalization
    (which may be challenging to understand through mathematical expressions, but becomes more accessible through code,
    这是相关的论文见：https://arxiv.org/abs/1502.03167#，其中的算法图很有用处).
    '''


    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):

        #初始化parameters
        self.input_size = input_size
        self.hidden_size_list  = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        #初始化weight
        self.__init_weight(weight_init_std)

        #生成layers
        activation_layer = {'sigmoid':SigmoidLayer,'relu':ReluLayer}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = AffineLayer(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx - 1])# 因为gamma和beta的维度和上一层的输出维度一致
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                         self.params['beta' + str(idx)])

            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = AffineLayer(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    #初始化weight和bias的具体实现
    def __init_weight(self, weight_init_std):
        '''Initialize the weights and biases of the network. '''
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # He initialization
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # Xavier initialization
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])


    #预测
    def predict(self, x, train_flg=False):
        '''Predict the output of the network.'''
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    #算loss
    def loss(self, x, t, train_flg=False):
        '''Calculate the loss of the network. 当然，y会用predict算出来'''
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    #算acc
    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    #最后是算梯度
    def gradient(self, x, t):

        # forword
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        # Loop through the layers in reverse order
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Store gradients in a dictionary
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params[
                'W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads


