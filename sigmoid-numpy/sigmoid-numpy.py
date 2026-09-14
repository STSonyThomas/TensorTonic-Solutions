import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    x_array = np.asarray(x, dtype=np.float64)
    result = 1/(1+np.exp(-x_array))
    return result.item() if result.ndim == 0 else result
    pass