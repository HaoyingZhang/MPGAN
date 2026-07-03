import sys, os
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance, kurtosis
from tslearn.metrics import dtw
from scipy.signal import find_peaks
from scipy.signal import welch as scipy_welch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # Add root directory to path

from features_extraction import ts_entropy

def normalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())

def pearson_correlation(x, y, threshold=None):
    """Compute the Pearson correlation between two time series x and y."""
    pcc = 0
    if np.all(x == y) or np.all(x == -y):
        pcc = 1.0
    elif np.all(x == x[0]) or np.all(y == y[0]):
        pcc = 0
    else:
        pcc = stats.pearsonr(x, y).statistic
        if pcc<0:
            pcc = -pcc
    if threshold is not None:
        pcc = (round(pcc,1) >= threshold)
    return pcc

def partial_pearson_correlation(x, y, m, epsilon=None):
    n_patterns = len(x) - m + 1
    pccs = []
    for i in range(n_patterns):
        pcc = pearson_correlation(x[i:i+m], y[i:i+m])
        pccs.append(pcc)
    pcc = np.max(pccs)
    if epsilon is not None:
        return round(pcc,1) >= epsilon
    else:
        return pcc
    
def mse(x, y):
    return ((x-y)**2).mean()

def rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.mean((x - y) ** 2))

def rmse_inv_check(list1, list2, epsilon=None):
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)

    n1 = normalize(list1)
    n2 = normalize(list2)
    n2_inv = normalize(-list2)

    rmse = np.min([np.sqrt(np.mean((n1 - n2) ** 2)), np.sqrt(np.mean((n1 - n2_inv) ** 2))])
    if epsilon is not None:
        return round(rmse, 1) <= epsilon
    else:
        return rmse

def partial_rmse(x, y, m, epsilon=None):
    n_patterns = len(x) - m + 1
    pccs = []
    for i in range(n_patterns):
        pcc = rmse_inv_check(x[i:i+m], y[i:i+m])
        pccs.append(pcc)
    rmse = np.min(pccs)
    if epsilon is not None:
        return round(rmse, 1)<=epsilon
    else:
        return rmse

def zdtw(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return dtw(a, b)

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

def mpi_distance(mpi_orig, mpi_pert):
    # n = len(mpi_orig)
    # positions = np.arange(n)

    # orig_weights = np.bincount(mpi_orig, minlength=n) / n
    # pert_weights = np.bincount(mpi_pert, minlength=n) / n

    # emd = wasserstein_distance(positions, positions, orig_weights, pert_weights)
    emd = rmse(mpi_orig, mpi_pert)
    return emd

def mpd_distance(mpd_orig, mpd_pert):
    return 1-pearson_correlation(mpd_orig, mpd_pert)

def detection_peak(ts_original, ts_reconstructed, threshold=None):
    fs = 100
    ts_original = np.asarray(ts_original)
    ts_reconstructed = np.asarray(ts_reconstructed)
    n = len(ts_original)
    pre_r  = int(0.20 * fs)   # 200 ms before R
    post_r = int(0.40 * fs)   # 400 ms after  R

    # find peaks in ts_original
    thresh   = ts_original.mean() + 0.4 * ts_original.std()
    min_dist = int(fs * 0.3)
    all_peaks_original, _ = find_peaks(ts_original, height=thresh,distance=min_dist, prominence=0.1)

    # Ignore the first or the last false peak detected
    valid_peaks_original = all_peaks_original[
        (all_peaks_original >= pre_r) & (all_peaks_original + post_r <= n)
    ]

    if len(valid_peaks_original) == 0:
        return 1

    # find peaks in ts_reconstructed
    thresh   = ts_reconstructed.mean() + 0.4 * ts_reconstructed.std()
    min_dist = int(fs * 0.3)
    all_peaks_reconstructed, _ = find_peaks(ts_reconstructed, height=thresh,distance=min_dist, prominence=0.1)
    valid_peaks_reconstructed = all_peaks_reconstructed[
        (all_peaks_reconstructed >= pre_r) & (all_peaks_reconstructed + post_r <= n)
    ]


    diff = len(valid_peaks_original) - len(valid_peaks_reconstructed)
    if diff < 0:
        return 1
    else:
        if threshold is None:
            return diff/len(valid_peaks_original)
        else:
            return diff/len(valid_peaks_original) > threshold

def cardiac_frequency(ts_original, ts_reconstructed, threshold=None):
    fs = 100
    pre_r  = int(0.20 * fs)   # 200 ms before R
    post_r = int(0.40 * fs)   # 400 ms after  R
    min_dist = 30
    n = len(ts_original)

    ts = np.asarray(ts_original)
    thresh = np.mean(ts) + 2.0 * ts.std()
    all_peaks_original, _ = find_peaks(ts, height=thresh,distance=min_dist, prominence=0.1)
    valid_peaks_original = all_peaks_original[
        (all_peaks_original >= pre_r) & (all_peaks_original + post_r <= n)
    ]
    

    ts = np.asarray(ts_reconstructed)
    thresh = np.mean(ts) + 2.0 * ts.std()
    all_peaks_reconstructed, _ = find_peaks(ts, height=thresh,distance=min_dist, prominence=0.1)
    valid_peaks_reconstructed = all_peaks_reconstructed[
        (all_peaks_reconstructed >= pre_r) & (all_peaks_reconstructed + post_r <= n)
    ]

    if len(valid_peaks_original) < 2 or len(valid_peaks_reconstructed) < 2:
        # print("invalid")
        return np.nan
    beat_len_reconstructed = np.mean([valid_peaks_reconstructed[i+1]-valid_peaks_reconstructed[i] for i in range(len(valid_peaks_reconstructed)-1)])
    beat_len_original = np.mean([valid_peaks_original[i+1]-valid_peaks_original[i] for i in range(len(valid_peaks_original)-1)])
    acc = 1 - np.abs(beat_len_reconstructed-beat_len_original)/beat_len_original
    if threshold is not None:
        return acc > threshold
    else:
        return acc
    
def eeg_features(ts):
    hist, _ = np.histogram(ts, bins=10)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    fs = 500
    n = len(ts)
    nperseg    = min(n, int(fs * 2))
    freqs, psd = scipy_welch(ts, fs=fs, nperseg=nperseg)
    total_pwr  = psd.sum() + 1e-12

    def _band(fmin, fmax):
        return float(psd[(freqs >= fmin) & (freqs < fmax)].sum() / total_pwr)

    delta = _band(0.5, 4)
    theta = _band(4, 8)
    alpha = _band(8, 13)
    beta  = _band(13, 30)
    kurt = kurtosis(ts)
    gamma = _band(30, min(fs / 2.0 - 1.0, 45.0))
    return {"kurtosis": kurt, "entropy": entropy, "delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma}

def delta_loss(ts_original, ts_reconstructed, threshold=None):
    eeg_features_original = eeg_features(ts_original)
    eeg_features_reconstructed = eeg_features(ts_reconstructed)
    return np.abs(eeg_features_reconstructed["delta"]-eeg_features_original["delta"])/eeg_features_original["delta"]

def alpha_loss(ts_original, ts_reconstructed, threshold=None):
    eeg_features_original = eeg_features(ts_original)
    eeg_features_reconstructed = eeg_features(ts_reconstructed)
    return np.abs(eeg_features_reconstructed["alpha"]-eeg_features_original["alpha"])/eeg_features_original["alpha"]

def entropy_loss(ts_original, ts_reconstructed, threshold=None):
    eeg_features_original = eeg_features(ts_original)
    eeg_features_reconstructed = eeg_features(ts_reconstructed)
    return np.abs(eeg_features_reconstructed["entropy"]-eeg_features_original["entropy"])/eeg_features_original["entropy"]

def kurtosis_loss(ts_original, ts_reconstructed, threshold=None):
    eeg_features_original = eeg_features(ts_original)
    eeg_features_reconstructed = eeg_features(ts_reconstructed)
    return np.abs(eeg_features_reconstructed["kurtosis"]-eeg_features_original["kurtosis"])/eeg_features_original["kurtosis"]