import numpy as np

#numerical gradient
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        #f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad

def numerical_gradient(f, X):
    #for X in 1d
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    #for X in more than 1d
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

#gradiant descent
def gradiant_descent(f,init_x,lr,step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, X)
        x -= lr*grad

    return x

