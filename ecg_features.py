from scipy.signal import find_peaks, welch as scipy_welch
import pywt

import numpy as np
import os, sys, json
from scipy import stats
import matplotlib
from dtaidistance import dtw
import wfdb
import torch
from src.utils_matrix_profile import normalize, build_mp_embedding
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from sklearn.metrics import roc_auc_score
from ecgdetectors import Detectors


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


def ts_entropy(ts, fs=150, epsilon=0.5):
    """
    Beat-aligned morphology entropy + transition entropy for ECG biometrics.

    Returns a dict with:
      - morphology_entropy  : normalized Shannon entropy over beat-type distribution
      - transition_entropy  : entropy of beat-to-beat transition matrix (Markov signature)
      - n_clusters          : number of distinct beat morphologies detected

    Beat segmentation uses R-peak detection so windows are cardiac-cycle aligned,
    avoiding the artifact of fixed-length windows splitting QRS complexes.
    Falls back to fixed-length segmentation if too few R-peaks are found.
    """
    ts = np.asarray(ts, dtype=np.float64)
    n  = len(ts)

    # --- R-peak detection ---
    thresh   = ts.mean() + 0.4 * ts.std()
    min_dist = int(fs * 0.3)
    peaks, _ = find_peaks(ts, height=thresh, distance=min_dist, prominence=0.1)

    pre_r  = int(0.20 * fs)
    post_r = int(0.40 * fs)
    valid  = peaks[(peaks >= pre_r) & (peaks + post_r <= n)]

    if len(valid) >= 3:
        segments = [ts[p - pre_r : p + post_r] for p in valid]
    else:
        # fallback: fixed-length windows
        m = int(0.6 * fs)
        if n < 2 * m:
            return {'morphology_entropy': float('nan'),
                    'transition_entropy': float('nan'),
                    'n_clusters': 0}
        segments = [ts[i*m:(i+1)*m] for i in range(n // m)]

    # --- Z-normalise each beat ---
    normed = []
    for seg in segments:
        std = seg.std()
        normed.append((seg - seg.mean()) / (std + 1e-8))

    # --- Greedy clustering ---
    prototypes, labels = [], []
    for a in normed:
        if not prototypes:
            prototypes.append(a)
            labels.append(0)
            continue
        dists = np.array([np.linalg.norm(a - p) for p in prototypes])
        j = int(np.argmin(dists))
        if dists[j] < epsilon:
            labels.append(j)
        else:
            labels.append(len(prototypes))
            prototypes.append(a)

    labels = np.array(labels)
    k      = len(prototypes)

    # --- Morphology entropy ---
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    probs  = counts / counts.sum()
    if k <= 1:
        morph_ent = 0.0
    else:
        morph_ent = float(scipy_entropy(probs) / np.log(k))

    # --- Transition entropy (beat i -> beat i+1) ---
    if len(labels) >= 2:
        trans = np.zeros((k, k), dtype=np.float64)
        for a, b in zip(labels[:-1], labels[1:]):
            trans[a, b] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans / row_sums
        # row-wise entropy, weighted by how often each state is visited
        state_freq = counts / counts.sum()
        trans_ent  = 0.0
        for i in range(k):
            row = trans_prob[i]
            row = row[row > 0]
            if len(row) > 1:
                trans_ent += state_freq[i] * float(-np.sum(row * np.log(row)) / np.log(k))
    else:
        trans_ent = 0.0

    return {
        'morphology_entropy': morph_ent,
        'transition_entropy': trans_ent,
        'n_clusters':         float(k),
    }

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

def extract_ecg_features(ts, fs=150):
    """
    General intrinsic ECG features for a single time series.

    Features
    --------
    kurtosis          — R-peaks create high excess kurtosis
    skewness          — R-peaks create positive skew
    std               — amplitude scale
    sample_entropy    — regularity / complexity (m=2, r=0.2·σ)
    dominant_freq     — should fall in HR band 0.5–3 Hz
    spectral_entropy  — low for periodic (ECG) signals
    psd_ratio         — fraction of power in 0.5–40 Hz ECG band
    autocorr_peak_lag — lag (s) of first autocorrelation peak (HR period)
    autocorr_peak_val — strength of that peak (high → periodic)
    """
    ts = np.asarray(ts, dtype=np.float64)
    n  = len(ts)

    # --- Sample entropy (m=2, Chebyshev template matching), measure of complexity  ---
    def _sample_entropy(x, m=2, r_coeff=0.2):
        r   = r_coeff * (np.std(x) + 1e-8)
        win_m  = np.lib.stride_tricks.sliding_window_view(x, m)      # (n-m+1, m)
        win_m1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)  # (n-m,   m+1)
        B, A = 0, 0
        for i in range(len(win_m1) - 1):
            B += int(np.sum(np.max(np.abs(win_m[i+1:]  - win_m[i]),  axis=1) < r))
            A += int(np.sum(np.max(np.abs(win_m1[i+1:] - win_m1[i]), axis=1) < r))
        return -np.log((A + 1e-8) / (B + 1e-8)) if B > 0 else 0.0

    # --- Time domain ---
    kurt_val  = float(kurtosis(ts))
    skew_val  = float(skew(ts))
    std_val   = float(np.std(ts))
    samp_ent  = _sample_entropy(ts)
    beat_ent  = ts_entropy(ts, fs=fs)

    # --- Frequency domain (Welch PSD) ---
    nperseg       = min(n, 256)
    freqs, psd    = scipy_welch(ts, fs=fs, nperseg=nperseg)
    psd_sum       = psd.sum() + 1e-12
    dominant_freq = float(freqs[np.argmax(psd[1:]) + 1])          # skip DC
    psd_norm      = psd / psd_sum
    spectral_ent  = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12))
                          / np.log(len(psd_norm) + 1e-12))
    psd_ratio     = float(psd[(freqs >= 0.5) & (freqs <= 40.0)].sum() / psd_sum)

    # --- Autocorrelation peak (first peak in HR lag range) ---
    ts_c = ts - ts.mean()
    ac   = np.correlate(ts_c, ts_c, mode='full')[n-1:]
    ac  /= (ac[0] + 1e-12)
    lag_min = max(1, int(fs * 0.25))            # 250 ms → 240 bpm max
    lag_max = min(len(ac) - 1, int(fs * 2.0))   # 2 s   →  30 bpm min
    if lag_min < lag_max:
        seg               = ac[lag_min:lag_max]
        pk                = int(np.argmax(seg))
        autocorr_peak_lag = float((lag_min + pk) / fs)
        autocorr_peak_val = float(seg[pk])
    else:
        autocorr_peak_lag, autocorr_peak_val = 0.0, 0.0
    
    # --- RR regulation ---
    rr = rr_regularity_score(ts, fs=fs)

    return {
        'kurtosis':             kurt_val,
        'skewness':             skew_val,
        'std':                  std_val,
        'sample_entropy':       samp_ent,
        'dominant_freq':        dominant_freq,
        'spectral_entropy':     spectral_ent,
        'psd_ratio':            psd_ratio,
        'autocorr_peak_lag':    autocorr_peak_lag,
        'autocorr_peak_val':    autocorr_peak_val,
        'rr_regulation':        rr,
        'morphology_entropy':   beat_ent['morphology_entropy'],
        'transition_entropy':   beat_ent['transition_entropy'],
        'n_beat_clusters':      beat_ent['n_clusters'],
    }


def extract_ecg_features_bis(ts, fs=150):
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
        baseline = template[: int(0.05 * beat_len)].mean()

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
        'pow_40p':              pow_40p,
        'qrs_band_ratio':       qrs_band_ratio,
        'autocorr_peak_lag':    autocorr_peak_lag,
        'autocorr_peak_val':    autocorr_peak_val,
        # --- new: fiducial amplitudes & ratios ---
        'r_amplitude':          r_amplitude,
        't_amplitude':          t_amplitude,
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