#cross entropy error based on mini-batch
import numpy as np
import  cupy as cp
def cross_entropy_error(y, t):
    # If t is an integer
    if isinstance(t, int):
        t = cp.array([t])
        y = y[cp.arange(y.shape[0]), t] # Use integer indexing to get y values
        return -cp.sum(cp.log(y + 1e-7)) / y.shape[0]

    # If t is a one-hot encoded array
    elif isinstance(t, cp.ndarray):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        return -cp.sum(t*cp.log(y + 1e-7))/batch_size

    else:
        raise TypeError("Invalid type for argument 't'. Expected int or numpy.ndarray.")