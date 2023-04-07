import numpy as np
from matplotlib import pyplot as plt

from twolayer.mnist import load_mnist
from twolayer.multi_layers import MultiLayerNet
from common.optimizer import *

#load
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Create the network
network = MultiLayerNet(input_size=784, hidden_size_list = [100], output_size=10, use_dropout=False,
                        use_batchnorm=False)

# Hyperparameters
iters_num = 20000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# some lists to store the loss and accuracy values
train_loss_list = []
train_acc_list = []
test_acc_list = []

# the number of iterations per epoch
iter_per_epoch = max(train_size / batch_size, 1)

# Train the network
for i in range(iters_num):
    # Get mini batch
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Calculate gradient
    grad = network.gradient(x_batch, t_batch)

    # Update parameters
    adam = Adam()
    adam.update(params=network.params, grads=network.params)

    # Record loss
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # Calculate accuracy
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Train acc, Test acc | " + str(train_acc) + ", " + str(test_acc))

# plot the accuracy
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
