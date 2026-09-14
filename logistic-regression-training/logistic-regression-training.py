import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here

    def _bce(y:np.ndarray,p:np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(p,eps,1-eps)
        loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        return float(loss)
        
    n_samples, n_features = X.shape
    W, b = np.zeros(n_features,dtype=np.float64), 0.0
    for _ in range(steps):
        z = X@W+b
        y_pred = _sigmoid(z)
        loss = _bce(y,y_pred)
        dL = y_pred - y
        dW = (X.T @ dL) / n_samples
        db = np.sum(dL) / n_samples

        W -= lr * dW
        b -= lr * db
    return W, b
        
    # pass