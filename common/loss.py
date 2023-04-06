#cross entropy error based on mini-batch
import numpy as np
def cross_entropy_error(y, t):
    # If t is an integer
    if isinstance(t, int):
        t = np.array([t])
        y = y[np.arange(y.shape[0]), t] # Use integer indexing to get y values
        return -np.sum(np.log(y + 1e-7)) / y.shape[0]

    # If t is a one-hot encoded array
    elif isinstance(t, np.ndarray):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        return -np.sum(t*np.log(y + 1e-7))/batch_size

    else:
        raise TypeError("Invalid type for argument 't'. Expected int or numpy.ndarray.")