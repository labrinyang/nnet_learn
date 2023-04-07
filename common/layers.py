import os
import sys

sys.path.append(os.pardir)
from common.function import *
from common.loss import *
import cupy as cp
import numpy as np

# MulLayer
class MulLayer:
    '''I am a layer used to da multiplication, please give me x y'''
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    '''I am Layer used to do addition, please give me x,y'''
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x+y

        return out

    def backward(self, dout):
        dx = dout
        dy = dout

        return dx, dy

# Relu layer
class ReluLayer:
    '''I am a layer used to do Relu, please give me x'''
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# sigmoid layer
class SigmoidLayer:
    '''I am a layer used to do sigmoid, please give me x'''
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+cp.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

# Affine layer
class AffineLayer:
    '''I am a layer used to do affine, please give me x, w, b'''

    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
       # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):

    # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = cp.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

# Softmax with loss layer
class SoftmaxWithLoss:
    '''I am a layer used to do softmax with loss, please give me x, t'''
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[cp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = cp.ones(1)  # 初始化为1的数组，避免乘法时的None!!!!!!!!!

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = cp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv layer's shape is 4D, Fully Connected layer's shape is 2D

        # Mean and variance used for testing
        self.running_mean = running_mean
        self.running_var = running_var

        # Intermediate data used during backward pass
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = cp.zeros(D)
            self.running_var = cp.zeros(D)

        self.batch_size = x.shape[0]

        if train_flg:
            mu = x.mean(axis=0)
            self.xc = x - mu
            var = cp.mean(self.xc ** 2, axis=0)
            self.std = cp.sqrt(var + 10e-7)
            self.xn = self.xc / self.std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.xc = x - self.running_mean
            self.std = cp.sqrt(self.running_var + 1e-7)
            self.xn = self.xc / self.std

        out = self.gamma * self.xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dgamma = cp.sum(self.xn * dout, axis=0).reshape(1, -1)
        dbeta = cp.sum(dout, axis=0).reshape(1, -1)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -cp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = cp.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

