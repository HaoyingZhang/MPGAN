import numpy as np
from scipy import stats

def normalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())

def pearson_correlation(x, y):
    """Compute the Pearson correlation between two time series x and y."""
    if np.all(x == y) or np.all(x == -y):
        return 1.0
    elif np.all(x == x[0]) or np.all(y == y[0]):
        return np.NaN
    return np.abs(stats.pearsonr(x, y).statistic)

def pearson_r2(y_pred, y_true, eps=1e-8):
    """
    Loss = 1 - r^2
    where r is the Pearson correlation between y_pred and y_true.

    Parameters
    ----------
    y_pred : list or np.ndarray
        Predicted time series
    y_true : list or np.ndarray
        Ground-truth time series
    eps : float
        Small constant for numerical stability

    Returns
    -------
    loss : float
        1 - Pearson correlation squared
    """

    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    # Center
    vx = y_pred - y_pred.mean()
    vy = y_true - y_true.mean()

    # Pearson correlation
    r_num = np.sum(vx * vy)
    r_den = np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + eps
    r = r_num / r_den

    return r ** 2

def rmse(list_1, list_2):
    return np.sqrt(np.mean((list_1  - list_2) ** 2))