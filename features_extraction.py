from scipy.signal import find_peaks, welch as scipy_welch, butter, filtfilt, hilbert
import pywt

import numpy as np
from scipy import stats
from dtaidistance import dtw
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from ecgdetectors import Detectors
import pandas as pd
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from scipy import stats
from scipy.optimize import curve_fit


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

def zdtw(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return dtw(a, b)


# def ts_entropy(ts, fs=150, epsilon=0.5):
#     """
#     Beat-aligned morphology entropy + transition entropy for ECG biometrics.

#     Returns a dict with:
#       - morphology_entropy  : normalized Shannon entropy over beat-type distribution
#       - transition_entropy  : entropy of beat-to-beat transition matrix (Markov signature)
#       - n_clusters          : number of distinct beat morphologies detected

#     Beat segmentation uses R-peak detection so windows are cardiac-cycle aligned,
#     avoiding the artifact of fixed-length windows splitting QRS complexes.
#     Falls back to fixed-length segmentation if too few R-peaks are found.
#     """
#     ts = np.asarray(ts, dtype=np.float64)
#     n  = len(ts)

#     # --- R-peak detection ---
#     thresh   = ts.mean() + 0.4 * ts.std()
#     min_dist = int(fs * 0.3)
#     peaks, _ = find_peaks(ts, height=thresh, distance=min_dist, prominence=0.1)

#     pre_r  = int(0.20 * fs)
#     post_r = int(0.40 * fs)
#     valid  = peaks[(peaks >= pre_r) & (peaks + post_r <= n)]

#     if len(valid) >= 3:
#         segments = [ts[p - pre_r : p + post_r] for p in valid]
#     else:
#         # fallback: fixed-length windows
#         m = int(0.6 * fs)
#         if n < 2 * m:
#             return {'morphology_entropy': float('nan'),
#                     'transition_entropy': float('nan'),
#                     'n_clusters': 0}
#         segments = [ts[i*m:(i+1)*m] for i in range(n // m)]

#     # --- Z-normalise each beat ---
#     normed = []
#     for seg in segments:
#         std = seg.std()
#         normed.append((seg - seg.mean()) / (std + 1e-8))

#     # --- Greedy clustering ---
#     prototypes, labels = [], []
#     for a in normed:
#         if not prototypes:
#             prototypes.append(a)
#             labels.append(0)
#             continue
#         dists = np.array([np.linalg.norm(a - p) for p in prototypes])
#         j = int(np.argmin(dists))
#         if dists[j] < epsilon:
#             labels.append(j)
#         else:
#             labels.append(len(prototypes))
#             prototypes.append(a)

#     labels = np.array(labels)
#     k      = len(prototypes)

#     # --- Morphology entropy ---
#     counts = np.bincount(labels, minlength=k).astype(np.float64)
#     probs  = counts / counts.sum()
#     if k <= 1:
#         morph_ent = 0.0
#     else:
#         morph_ent = float(scipy_entropy(probs) / np.log(k))

#     # --- Transition entropy (beat i -> beat i+1) ---
#     if len(labels) >= 2:
#         trans = np.zeros((k, k), dtype=np.float64)
#         for a, b in zip(labels[:-1], labels[1:]):
#             trans[a, b] += 1
#         row_sums = trans.sum(axis=1, keepdims=True)
#         row_sums[row_sums == 0] = 1
#         trans_prob = trans / row_sums
#         # row-wise entropy, weighted by how often each state is visited
#         state_freq = counts / counts.sum()
#         trans_ent  = 0.0
#         for i in range(k):
#             row = trans_prob[i]
#             row = row[row > 0]
#             if len(row) > 1:
#                 trans_ent += state_freq[i] * float(-np.sum(row * np.log(row)) / np.log(k))
#     else:
#         trans_ent = 0.0

#     return {
#         'morphology_entropy': morph_ent,
#         'transition_entropy': trans_ent,
#         'n_clusters':         float(k),
#     }

def rr_regularity_score(ts, fs=150, min_gap=100):
    """
    Score based on the regularity of RR intervals detected by Pan-Tompkins.

    Steps:
      1. Detect R peaks with Pan-Tompkins.
      2. Compute RR intervals (consecutive peak differences in samples).
      3. Keep only intervals >= min_gap samples.
      4. Score = 1 - CV  (coefficient of variation), clipped to [0, 1].
         A perfectly regular rhythm → CV=0 → score=1.
         High variability or too few valid peaks → score=0.

    Parameters
    ----------
    ts      : 1-D array-like ECG signal
    fs      : sampling frequency in Hz (default 150)
    min_gap : minimum RR interval in samples to retain (default 100)

    Returns
    -------
    float in [0, 1]  (higher = more regular RR distribution)
    """
    ts = np.asarray(ts, dtype=np.float64)
    try:
        detectors = Detectors(fs)
        r_peaks = np.array(detectors.pan_tompkins_detector(ts), dtype=int)
    except Exception:
        return 0.0

    if len(r_peaks) < 2:
        return 0.0

    rr = np.diff(r_peaks)
    rr = rr[rr >= min_gap]

    if len(rr) < 2:
        return 0.0

    cv = rr.std() / (rr.mean() + 1e-8)
    return float(np.clip(1.0 - cv, 0.0, 1.0))

def ts_to_tsfresh_df(ts, ts_id=0):
    return pd.DataFrame({
        'id':    ts_id,
        'time':  np.arange(len(ts)),
        'value': np.asarray(ts, dtype=np.float64),
    })


def ts_entropy(ts, m=100, epsilon=3.0):
    """
    Subsequence-level pattern entropy via epsilon-clustering on z-normalized
    Euclidean distance. Returns normalized Shannon entropy in [0, 1].
    0 = perfectly regular (all beats identical), 1 = maximum disorder.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 2 * m:
        return float('nan')

    nb_patterns = len(ts) // m
    prototypes, counts = [], []

    for i in range(nb_patterns):
        current = ts[i*m:(i+1)*m]
        a = (current - current.mean()) / (current.std() + 1e-8)

        if not prototypes:
            prototypes.append(a)
            counts.append(1)
            continue

        dists = np.array([np.linalg.norm(a - p) for p in prototypes])
        j = np.argmin(dists)
        if dists[j] < epsilon:
            counts[j] += 1
        else:
            prototypes.append(a)
            counts.append(1)

    counts = np.asarray(counts, dtype=np.float64)
    probs  = counts / counts.sum()
    if len(probs) <= 1:
        return 0.0
    return float(scipy_entropy(probs) / np.log(len(probs)))

_ROBUST_EEG_KEYS = {
    'iaf', 'entropy', 'spectral_entropy', 'offset', 'mean',
    'total_power_log', 'exponent', 'hjorth_complexity',
    'alpha_bandwidth', 'gamma_env_mean',
}

def extract_eeg_features(ts, fs=500, robust=False):
    """
    Single-channel EEG features for biometric reidentification.

    Band powers use a 4th-order Butterworth filter-bank (accurate on short
    segments regardless of length).  Features that need a frequency axis
    (IAF, SEF95, FWP, aperiodic) use zero-padded Welch for fine bin spacing.
    Features chosen for cross-session stability (Palaniappan & Mandic 2007,
    La Rocca et al. 2014, Maiorana et al. 2016).
    """
    ts = np.asarray(ts, dtype=np.float64)
    n  = len(ts)
    nyq = fs / 2.0

    # FFT-based features (frequency domain)
    fft = np.abs(np.fft.fft(ts))
    low_freq = np.sum(fft[:len(fft)//4])
    mid_freq = np.sum(fft[len(fft)//4:len(fft)//2])
    peak_freq = np.max(fft)
    peak_freq_ind = np.argmax(fft)

    # Entropy-based feature
    hist, _ = np.histogram(ts, bins=10)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-10))

    # Zero crossing rate
    x_centered = ts - np.mean(ts)
    zeros = np.sum(np.abs(np.diff(np.sign(x_centered)))) / (2 * len(ts))

    # Autocorrelation at lag 1
    acf = np.correlate(x_centered, x_centered, mode='full')
    lag1_corr = acf[len(acf)//2 + 1] / (acf[len(acf)//2] + 1e-10)

    # --- Filter-bank: one pass per band gives power + Hilbert envelope stats ---
    total_var = float(np.var(ts)) + 1e-12

    def _band_features(lo, hi):
        hi_c = min(hi, nyq - 0.5)
        if lo >= hi_c:
            return 0.0, 0.0, 0.0
        b, a = butter(4, [lo / nyq, hi_c / nyq], btype='band')
        filtered  = filtfilt(b, a, ts)
        power     = float(np.var(filtered)) / total_var
        envelope  = np.abs(hilbert(filtered))
        env_mean  = float(envelope.mean())
        env_var   = float(envelope.var()) / total_var
        return power, env_mean, env_var

    delta, delta_env_mean, delta_env_var = _band_features(0.5, 4)
    theta, theta_env_mean, theta_env_var = _band_features(4,   8)
    alpha, alpha_env_mean, alpha_env_var = _band_features(8,  13)
    beta,  beta_env_mean,  beta_env_var  = _band_features(13, 30)
    gamma, gamma_env_mean, gamma_env_var = _band_features(30, min(nyq - 0.5, 45.0))

    # --- Zero-padded Welch for features that need a frequency axis ---
    # nperseg = n uses the full segment; nfft zero-pads for ~0.1 Hz bin spacing
    nfft = max(4096, 2 * n)
    freqs, psd = scipy_welch(ts, fs=fs, nperseg=n, nfft=nfft)
    total_pwr  = psd.sum()

    # Frequency-Weighted Power (Monsy et al., IET Biometrics 2020)
    total_fwp = float((freqs * psd).sum()) + 1e-12
    def _fwp(fmin, fmax):
        mask = (freqs >= fmin) & (freqs < fmax)
        return float((freqs[mask] * psd[mask]).sum() / total_fwp)

    fwp_delta = _fwp(0.5, 4)
    fwp_theta = _fwp(4, 8)
    fwp_alpha = _fwp(8, 13)
    fwp_beta  = _fwp(13, 30)
    fwp_gamma = _fwp(30, min(nyq - 0.5, 45.0))

    # Individual Alpha Frequency — peak within 8–13 Hz
    a_mask = (freqs >= 8) & (freqs <= 13)
    if a_mask.any():
        a_psd   = psd[a_mask]
        a_freqs = freqs[a_mask]
        iaf     = float(a_freqs[np.argmax(a_psd)])
        above   = a_psd >= a_psd.max() / 2.0
        alpha_bw = float(a_freqs[above].max() - a_freqs[above].min()) if above.sum() >= 2 else 0.0
    else:
        iaf, alpha_bw = 10.0, 0.0

    # Spectral edge frequency: frequency below which 95% of power lies
    cumpow  = np.cumsum(psd)
    sef_idx = np.searchsorted(cumpow, 0.95 * cumpow[-1])
    sef95   = float(freqs[min(sef_idx, len(freqs) - 1)])

    # Normalised spectral entropy
    psd_norm     = psd / total_pwr
    spectral_ent = float(
        -np.sum(psd_norm * np.log(psd_norm + 1e-12)) / np.log(len(psd_norm) + 1e-12)
    )

    # Hjorth parameters — mobility and complexity are stable EEG identity cues
    dx   = np.diff(ts)
    ddx  = np.diff(dx)
    var_x   = np.var(ts)   + 1e-12
    var_dx  = np.var(dx)   + 1e-12
    var_ddx = np.var(ddx)  + 1e-12
    hjorth_mobility   = float(np.sqrt(var_dx / var_x))
    hjorth_complexity = float(np.sqrt(var_ddx / var_dx) / (hjorth_mobility + 1e-12))

    kurt = kurtosis(ts)
    mean = np.mean(ts)
    std  = np.std(ts)

    aperiodic = fit_aperiodic(freqs, psd)

    # --- Higuchi Fractal Dimension ---
    def _higuchi_fd(x, kmax=8):
        n = len(x)
        L = []
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(1, k + 1):
                count = (n - m) // k
                if count < 1:
                    continue
                idx = m - 1 + np.arange(1, count + 1) * k
                Lmk = np.sum(np.abs(x[idx] - x[idx - k])) * (n - 1) / (count * k * k)
                Lk.append(Lmk)
            if Lk:
                L.append(np.mean(Lk))
        if len(L) < 2:
            return 1.5
        slope, _ = np.polyfit(np.log(np.arange(1, len(L) + 1)),
                              np.log(np.array(L) + 1e-12), 1)
        return float(-slope)

    hfd = _higuchi_fd(ts)

    # --- AR coefficients via Yule-Walker (order 8) ---
    _AR_ORDER = 8

    def _ar_coeffs(x, order=_AR_ORDER):
        x = x - x.mean()
        n = len(x)
        acf = np.array([np.dot(x[:n - k], x[k:]) / (n - k) for k in range(order + 1)])
        R = np.array([[acf[abs(i - j)] for j in range(order)] for i in range(order)])
        try:
            return np.linalg.solve(R, acf[1:order + 1])
        except np.linalg.LinAlgError:
            return np.zeros(order)

    ar = _ar_coeffs(ts)

    all_feats = {
    # 'kurtosis':           kurt,
    'mean':               mean,
    'std':                std,
    'entropy':            entropy,
    'delta_power':        delta,
    'theta_power':        theta,
    'alpha_power':        alpha,
    'beta_power':         beta,
    'gamma_power':        gamma,
    'iaf':                iaf,
    'alpha_bandwidth':    alpha_bw,
    'sef95':              sef95,
    # 'theta_alpha_ratio':  float(theta / (alpha + 1e-8)),
    # 'delta_alpha_ratio':  float(delta / (alpha + 1e-8)),
    # 'alpha_beta_ratio':   float(alpha / (beta  + 1e-8)),
    'spectral_entropy':   spectral_ent,
    # 'total_power_log':    float(np.log(total_pwr)),
    'hjorth_mobility':    hjorth_mobility,
    'hjorth_complexity':  hjorth_complexity,
    'offset':             aperiodic["offset"],
    'exponent':           aperiodic["exponent"],
    #Frequency-Weighted Power (Monsy et al., IET Biometrics 2020)
    'fwp_delta':          fwp_delta,
    'fwp_theta':          fwp_theta,
    'fwp_alpha':          fwp_alpha,
    'fwp_beta':           fwp_beta,
    'fwp_gamma':          fwp_gamma,
    # Higuchi Fractal Dimension
    'hfd':                hfd,
    # AR(8) coefficients via Yule-Walker
    'ar_1':               float(ar[0]),
    'ar_2':               float(ar[1]),
    # 'ar_3':               float(ar[2]),
    # 'ar_4':               float(ar[3]),
    # 'ar_5':               float(ar[4]),
    # 'ar_6':               float(ar[5]),
    # 'ar_7':               float(ar[6]),
    # 'ar_8':               float(ar[7]),
    # Hilbert envelope stats per band (mean amplitude + normalised variance)
    'delta_env_mean':     delta_env_mean,
    'delta_env_var':      delta_env_var,
    'theta_env_mean':     theta_env_mean,
    'theta_env_var':      theta_env_var,
    'alpha_env_mean':     alpha_env_mean,
    'alpha_env_var':      alpha_env_var,
    'beta_env_mean':      beta_env_mean,
    'beta_env_var':       beta_env_var,
    'gamma_env_mean':     gamma_env_mean,
    'gamma_env_var':      gamma_env_var,

    }
    if robust:
        return {k: v for k, v in all_feats.items() if k in _ROBUST_EEG_KEYS}
    return all_feats


def extract_eeg_features_bis(ts, fs=500):
    # FFT-based features (frequency domain)
    fft = np.abs(np.fft.fft(ts))
    low_freq = np.sum(fft[:len(fft)//4])
    mid_freq = np.sum(fft[len(fft)//4:len(fft)//2])
    peak_freq = np.max(fft)
    peak_freq_ind = np.argmax(fft)
    
    # Entropy-based feature
    hist, _ = np.histogram(ts, bins=10)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-10))

    # Zero crossing rate
    x_centered = ts - np.mean(ts)
    zeros = np.sum(np.abs(np.diff(np.sign(x_centered)))) / (2 * len(ts))

    # Autocorrelation at lag 1
    acf = np.correlate(x_centered, x_centered, mode='full')
    lag1_corr = acf[len(acf)//2 + 1] / (acf[len(acf)//2] + 1e-10)

    return {
        "low_freq": low_freq,
        "mid_freq": mid_freq,
        "peak_freq": peak_freq,
        "peak_freq_ind": peak_freq_ind,
        "entropy": entropy,
        "zeros": zeros,
        "lag1_corr": lag1_corr
    }

def aperiodic_model(log_freqs, offset, exponent):
    """Linear model in log-log space: log P = offset - exponent * log f"""
    return offset - exponent * log_freqs

def fit_aperiodic(freqs, psd, freq_range=(1, 40)):
    """
    Fit the aperiodic 1/f component and return parameters + R².
    
    freqs     : frequency array (Hz)
    psd       : power spectral density array
    freq_range: frequency range to fit over
    """
    # Select frequency range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]
    
    # Work in log-log space
    log_freqs = np.log10(freqs_fit)
    log_psd = np.log10(psd_fit)
    
    # Fit linear model (offset + slope)
    params, _ = curve_fit(aperiodic_model, log_freqs, log_psd,
                          p0=[1.0, 1.0], maxfev=5000)
    offset, exponent = params
    
    # Predicted values
    log_psd_hat = aperiodic_model(log_freqs, offset, exponent)
    
    # R² in log-log space
    ss_res = np.sum((log_psd - log_psd_hat) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        "offset": offset,
        "exponent": exponent,
        "r2": r2,
        "fitted_psd": 10 ** log_psd_hat,
        "freqs": freqs_fit
    }




def extract_ecg_features(ts, fs=150):
    """
    Morphology-focused ECG features for person re-identification.

    Improvements over the original
    --------------------------------
    - Beat template extracted from valid R-peaks (edge beats rejected)
    - Fiducial amplitudes & ratios  : r_amplitude, t_amplitude, s_amplitude,
                                      rt_ratio, rs_ratio  (normalise out session
                                      amplitude drift)
    - QRS geometry                  : qrs_width_ms, qrs_area, st_level
    - T-wave energy                 : t_area
    - Template shape (cosine basis) : cosine_1 … cosine_8  (encode morphology
                                      without needing PCA across subjects)
    - Morphology consistency        : beat-to-template correlation (intra-session
                                      stability check)
    - Sub-band PSD ratios           : pow_0_5, pow_0p5_5, pow_5_15, pow_15_40,
                                      pow_40p, qrs_band_ratio
                                      (QRS band 5–15 Hz is most discriminative)

    Kept from original (stable enough or useful as secondary cues)
    --------------------------------------------------------------
    - kurtosis, skewness, std       : retained but de-weighted in SVM
    - spectral_entropy              : kept
    - autocorr_peak_lag/val         : kept, now computed on edge-cleaned signal
    - sample_entropy                : kept

    Removed
    -------
    - dominant_freq   : varies ±0.5 Hz intra-person across sessions (HR drift)
    - psd_ratio       : replaced by finer sub-band ratios
    - rr_regulation   : only 3–4 intervals on a 3 s window → too noisy
    - subsequence_entropy, pattern_entropy : external deps (ts_entropy); fragile
                                             on short windows

    Parameters
    ----------
    ts : array-like   1-D ECG time series
    fs : int          Sampling frequency in Hz (default 150)

    Returns
    -------
    dict of float features
    """
    ts = np.asarray(ts, dtype=np.float64)
    n  = len(ts)

    # ------------------------------------------------------------------ #
    # 0. R-peak detection with edge rejection
    # ------------------------------------------------------------------ #
    # median = np.median(ts)
    # mad    = np.median(np.abs(ts - median))  # median absolute deviation, robust std

    # thresh   = median + 2.0 * mad
    thresh = np.mean(ts) + 2.0 * ts.std()
    min_dist = 30

    # all_peaks, _ = find_peaks(ts,height=thresh,distance=min_dist,prominence=0.3, width=(1, 15))  
    all_peaks, _ = find_peaks(ts, height=thresh,distance=min_dist, prominence=0.1)
    pre_r  = int(0.20 * fs)   # 200 ms before R
    post_r = int(0.40 * fs)   # 400 ms after  R
    beat_len = pre_r + post_r

    # Ignore the first or the last false peak detected
    valid_peaks = all_peaks[
        (all_peaks >= pre_r) & (all_peaks + post_r <= n)
    ]
    if len(valid_peaks)>1:
        beat_len = np.mean([valid_peaks[i+1]-valid_peaks[i] for i in range(len(valid_peaks)-1)])
        pre_r = int(0.2 * beat_len)
        post_r = int(0.4 * beat_len)
        valid_peaks = valid_peaks[
            (valid_peaks >= pre_r) & (valid_peaks + post_r <= n)
        ]

    # ------------------------------------------------------------------ #
    # 1. Sample entropy  (m=2, Chebyshev)
    # ------------------------------------------------------------------ #
    def _sample_entropy(x, m=2, r_coeff=0.2):
        r      = r_coeff * (np.std(x) + 1e-8)
        win_m  = np.lib.stride_tricks.sliding_window_view(x, m)
        win_m1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)
        B, A   = 0, 0
        for i in range(len(win_m1) - 1):
            B += int(np.sum(np.max(np.abs(win_m[i+1:]  - win_m[i]),  axis=1) < r))
            A += int(np.sum(np.max(np.abs(win_m1[i+1:] - win_m1[i]), axis=1) < r))
        return -np.log((A + 1e-8) / (B + 1e-8)) if B > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 2. Time-domain (original, kept)
    # ------------------------------------------------------------------ #
    kurt_val = float(kurtosis(ts))
    skew_val = float(skew(ts))
    std_val  = float(np.std(ts))
    samp_ent  = _sample_entropy(ts)

    # ------------------------------------------------------------------ #
    # 3. Frequency domain — sub-band PSD ratios
    # Using Welch instead of FT because we have noisy signals
    # ------------------------------------------------------------------ #
    nperseg = min(n, 256)
    freqs, psd = scipy_welch(ts, fs=fs, nperseg=nperseg)
    total_pow  = psd.sum() + 1e-12

    def _band_pow(lo, hi):
        return float(psd[(freqs >= lo) & (freqs < hi)].sum() / total_pow)

    pow_0_5    = _band_pow(0,    0.5)   # baseline drift
    pow_0p5_5  = _band_pow(0.5,  5.0)  # HR + P/T waves
    pow_5_15   = _band_pow(5.0,  15.0) # QRS — most person-discriminative
    pow_15_40  = _band_pow(15.0, 40.0) # high-freq QRS notches
    pow_40p    = _band_pow(40.0, fs/2)  # noise / EMG

    # QRS sharpness: ratio of QRS band to sub-Hz power
    qrs_band_ratio = pow_5_15 / (pow_0p5_5 + 1e-8)

    psd_norm      = psd / total_pow
    spectral_ent  = float(
        -np.sum(psd_norm * np.log(psd_norm + 1e-12))
        / np.log(len(psd_norm) + 1e-12)
    )
    dominant_freq = np.argmax(psd)

    # ------------------------------------------------------------------ #
    # 4. Autocorrelation peak  (on valid-peak-trimmed signal)
    # ------------------------------------------------------------------ #
    ts_c = ts - ts.mean()
    ac   = np.correlate(ts_c, ts_c, mode='full')[n-1:]
    ac  /= (ac[0] + 1e-12)
    lag_min = max(1, int(fs * 0.25))
    lag_max = min(len(ac) - 1, int(fs * 2.0)) # restrict the search window to 0.25 to 2 s
    if lag_min < lag_max:
        seg               = ac[lag_min:lag_max]
        pk                = int(np.argmax(seg))
        autocorr_peak_lag = float((lag_min + pk) / fs)
        autocorr_peak_val = float(seg[pk])
    else:
        autocorr_peak_lag, autocorr_peak_val = 0.0, 0.0

    # ------------------------------------------------------------------ #
    # 5. Beat-template morphology features
    # ------------------------------------------------------------------ #
    # defaults (used when too few valid beats)
    r_amplitude          = 0.0
    t_amplitude          = 0.0
    s_amplitude          = 0.0
    rt_ratio             = 0.0
    rs_ratio             = 0.0
    qrs_width_ms         = 0.0
    qrs_area             = 0.0
    t_area               = 0.0
    st_level             = 0.0
    morphology_consist   = 0.0
    cosine_coeffs        = [0.0] * 8

    if len(valid_peaks) >= 2:
        beats    = np.array([ts[pk - pre_r : pk + post_r] for pk in valid_peaks])
        template = beats.mean(axis=0)

        # Baseline: mean of first 50 ms
        baseline_win = template[: max(1, int(0.05 * beat_len))]
        baseline = baseline_win.mean()

        # R amplitude
        r_amp = np.max(template)

        # S-wave: minimum in first 40 ms after R
        s_win = template[pre_r : pre_r + int(0.04 * fs) + 1]
        s_amp = max(0.0, baseline - s_win.min())

        # T-wave: maximum in 150–380 ms after R
        t_start = pre_r + int(0.15 * fs)
        t_end   = pre_r + int(0.38 * fs)
        t_win   = template[t_start:t_end]
        t_amp   = max(0.0, t_win.max() - baseline) if len(t_win) > 0 else 0.0

        r_amplitude  = float(r_amp)
        t_amplitude  = float(t_amp)
        s_amplitude  = float(s_amp)
        rt_ratio     = float(t_amp  / (r_amp + 1e-8))
        rs_ratio     = float(s_amp  / (r_amp + 1e-8))

        # QRS width at 50 % of R height
        qrs_win  = template[pre_r - int(0.05*fs) : pre_r + int(0.06*fs)]
        half_r   = r_amp * 0.5 + baseline
        qrs_width_ms = float((qrs_win > half_r).sum() * 1000.0 / fs)

        # QRS and T-wave areas (absolute deflection from baseline)
        qrs_seg  = template[pre_r - int(0.04*fs) : pre_r + int(0.05*fs)]
        qrs_area = float(np.trapezoid(np.abs(qrs_seg - baseline)))

        t_seg    = template[t_start:t_end]
        t_area   = float(np.trapezoid(np.abs(t_seg - baseline))) if len(t_seg) > 1 else 0.0

        # ST level: 60 ms post R relative to baseline
        st_idx   = pre_r + int(0.06 * fs)
        st_level = float(template[st_idx] - baseline)

        # Template shape: project normalised template onto cosine basis
        t_norm = (template - template.mean()) / (template.std() + 1e-8)
        cosine_coeffs = [
            float(np.dot(t_norm, np.cos(np.pi * k * np.arange(pre_r + post_r) / (pre_r + post_r))) / (pre_r + post_r))
            for k in range(1, 9)
        ]

        # Beat-to-template morphology consistency
        t_norm_fix = (template - template.mean()) / (template.std() + 1e-8)
        corrs = []
        for b in beats:
            b_n = (b - b.mean()) / (b.std() + 1e-8)
            corrs.append(float(np.dot(b_n, t_norm_fix) / beat_len))
        morphology_consist = float(np.mean(corrs))

    # ------------------------------------------------------------------ #
    # 6. Wavelet features  (db4, 4-level DWT on the normalized beat template)
    #
    # Computed on the z-normalised beat *template* rather than the raw signal.
    # Template averaging suppresses IPOPT reconstruction artifacts and amplitude
    # drift, so only person-specific morphology survives.
    # d1/d2 are skipped: at fs~100-150 Hz they sit in the noise/EMG band and
    # carry no stable identity information in reconstructed signals.
    #
    # Template-relative band mapping (fs ≈ 100-150 Hz, beat_len ≈ 60-90 smp):
    #   d3: QRS complex 
    #   d4: QRS tail + T-wave onset
    #   approximation cA4: slow P/T-wave baseline shape
    # ------------------------------------------------------------------ #
    if len(valid_peaks) >= 2:
        # t_norm is already the z-normalised template from section 5
        wav_signal = t_norm
    else:
        # fallback: z-normalise the raw signal
        wav_signal = (ts - ts.mean()) / (ts.std() + 1e-8)

    wav_level = min(4, pywt.dwt_max_level(len(wav_signal), 'db4'))
    wav_coeffs = pywt.wavedec(wav_signal, wavelet='db4', level=wav_level)
    # wav_coeffs = [cAn, cDn, ..., cD1]  (n = wav_level)
    # Use only approx + two lowest-frequency detail bands — skip d1, d2 (noise)
    cA4 = wav_coeffs[0]
    cD4 = wav_coeffs[1] if len(wav_coeffs) > 1 else np.zeros(1)
    cD3 = wav_coeffs[2] if len(wav_coeffs) > 2 else np.zeros(1)
    e_approx = float(np.sum(cA4 ** 2))
    e_d4     = float(np.sum(cD4 ** 2))
    e_d3     = float(np.sum(cD3 ** 2))
    total_wav_e = e_approx + e_d4 + e_d3 + 1e-12

    wav_e_approx = e_approx / total_wav_e   # slow P/T-wave baseline shape
    wav_e_d4     = e_d4     / total_wav_e   # QRS tail + T-wave onset
    wav_e_d3     = e_d3     / total_wav_e   # QRS complex

    # Shannon entropy over the three retained components
    _p = np.array([wav_e_approx, wav_e_d4, wav_e_d3])
    wavelet_entropy  = float(-np.sum(_p * np.log(_p + 1e-12)))
    # QRS sharpness: how much energy is concentrated in d3 vs the slow bands
    wavelet_qrs_ratio = float(wav_e_d3 / (wav_e_approx + wav_e_d4 + 1e-8))

    return {
        # --- original time-domain ---
        'kurtosis':             kurt_val,
        'skewness':             skew_val,
        'std':                  std_val,
        'sample_entropy':       samp_ent,
        'bpm':                  beat_len,
        # --- original frequency-domain (refined) ---
        'spectral_entropy':     spectral_ent,
        'dominant_frequency':   dominant_freq,
        'pow_0_5':              pow_0_5,
        'pow_0p5_5':            pow_0p5_5,
        'pow_5_15':             pow_5_15,
        'pow_15_40':            pow_15_40,
        # 'pow_40p':              pow_40p,
        'qrs_band_ratio':       qrs_band_ratio,
        'autocorr_peak_lag':    autocorr_peak_lag,
        'autocorr_peak_val':    autocorr_peak_val,
        # --- new: fiducial amplitudes & ratios ---
        'r_amplitude':          r_amplitude,
        # 't_amplitude':          t_amplitude,
        # 's_amplitude':          s_amplitude,
        # 'rt_ratio':             rt_ratio,
        # 'rs_ratio':             rs_ratio,
        # # --- new: QRS geometry ---
        'qrs_width_ms':         qrs_width_ms,
        'qrs_area':             qrs_area,
        # 'st_level':             st_level,
        # # --- new: T-wave ---
        # 't_area':               t_area,
        # # --- new: template shape (cosine projection) ---
        'cosine_1':             cosine_coeffs[0],
        'cosine_2':             cosine_coeffs[1],
        'cosine_3':             cosine_coeffs[2],
        'cosine_4':             cosine_coeffs[3],
        'cosine_5':             cosine_coeffs[4],
        'cosine_6':             cosine_coeffs[5],
        'cosine_7':             cosine_coeffs[6],
        'cosine_8':             cosine_coeffs[7],
        # # --- new: morphology consistency ---
        # 'morphology_consistency': morphology_consist,
        # --- wavelet features (db4 on z-normalised beat template) ---
        # 'wav_e_approx':         wav_e_approx,   # slow P/T-wave baseline shape
        'wav_e_d4':             wav_e_d4,        # QRS tail + T-wave onset
        # 'wav_e_d3':             wav_e_d3,        # QRS complex energy fraction
        # 'wavelet_entropy':      wavelet_entropy,
        'wavelet_qrs_ratio':    wavelet_qrs_ratio,
    }

