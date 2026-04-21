from scipy.signal import welch as scipy_welch
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from ecgdetectors import Detectors
from ecg_features import extract_ecg_features_bis
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import stumpy


def ts_to_tsfresh_df(ts, ts_id=0):
    return pd.DataFrame({
        'id':    ts_id,
        'time':  np.arange(len(ts)),
        'value': np.asarray(ts, dtype=np.float64),
    })

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


def ecg_singling_score(ts, fs=150):
    """
    Phase 1 — how ECG-like is ts?

    Each criterion is a soft sigmoid gate based on established ECG physiology.
    Returns the geometric mean of all gates, in (0, 1].  Higher = more ECG-like.

    Gate centres / scales are justified as follows:

    kurtosis (Fisher excess kurtosis, scipy.stats.kurtosis default):
      Zhao & Zhang (Front. Physiol. 2018) and Orphanidou et al. (J. R. Soc.
      Interface 2022) define a noise-free ECG as having Pearson kurtosis > 5,
      i.e. Fisher excess kurtosis > 2 (Fisher = Pearson − 3).
      Center = 2.0; scale = 1.0 for a gradual transition.

    psd_ratio (fraction of power in 0.5–40 Hz):
      ANSI/AAMI EC13:2002 (cardiac monitoring standard, not EC11 which covers
      diagnostic devices at 0.05–150 Hz) specifies 0.5–40 Hz as the monitoring
      bandwidth.  A genuine ECG should have ≥ 80 % of its power in this band.
      Center = 0.80, scale = 10.

    autocorr_peak_val (normalised autocorrelation at heartbeat lag):
      Heuristic: clear quasi-periodicity at the RR interval is assumed to
      produce a normalised autocorrelation peak > 0.30.  No published
      quantitative threshold was found; this value requires empirical
      validation on your dataset.

    spectral_entropy (normalised, 0 = single tone, 1 = white noise):
      Heuristic: quasi-periodic ECG is less spectrally diffuse than noise.
      Center = 0.80 is a conservative cut-off; no published reference for
      this specific threshold was found — treat as a tunable parameter.

    dominant_freq in [0.5, 3.0] Hz:
      Physiological HR range 30–180 bpm corresponds exactly to 0.5–3.0 Hz
      (Task Force of ESC/NASPE, Circulation 1996;93:1043-1065).
      Two gates create a soft band-pass at the exact physiological limits.
    """
    def _sig(x, center, scale=5.0):
        return 1.0 / (1.0 + np.exp(-scale * (x - center)))

    f = extract_ecg_features_bis(ts, fs)
    gates = [
        _sig(f['kurtosis'],           center=3.5,  scale=2.0),        # Fisher excess kurtosis > 2 ↔ Pearson > 5 (noise-free ECG threshold)
        # _sig(f['psd_ratio'],          center=0.80, scale=10.0),        # ≥80 % power in AHA 0.5–40 Hz band
        _sig(f['autocorr_peak_lag'],  center=0.3, scale=18.5),        # clear heartbeat periodicity
        # 1.0 - _sig(f['spectral_entropy'], center=0.50, scale=10.0),   # quasi-periodic, not noise-like
        # dominant freq strictly within physiological HR band [0.5, 3.0] Hz
        # _sig(f['dominant_freq'], center=0.5, scale=10.0) * (1.0 - _sig(f['dominant_freq'], center=3.0, scale=10.0)),
        1.0 - _sig(f['subsequence_entropy'], center=0.38, scale=2.12), # regular ECG has low pattern entropy
        rr_regularity_score(ts, fs),                                    # Pan-Tompkins RR regularity (CV-based)
    ]
    return float(np.prod(gates) ** (1.0 / len(gates)))


def extract_personalized_features(ts, reference_pool, fs=150):
    """
    Phase 2 — compare ts against the victim's reference ECG pool.

    reference_pool: list of 1-D np.ndarray — non-overlapping ECGs of the victim.

    Returns a dict with:
      mean_pcc / max_pcc        — Pearson similarity to pool
      mean_dtw / min_dtw        — z-DTW distance to pool (lower = closer)
      kurtosis_zscore           — z-score vs victim's kurtosis distribution
      entropy_zscore            — z-score vs victim's sample entropy
      dominant_freq_zscore      — z-score vs victim's dominant frequency
      mahalanobis_dist          — distance in full feature space from victim centroid
      mean_abs_zscore           — mean absolute z-score across all features
    """
    ts  = np.asarray(ts, dtype=np.float64)
    ref = [np.asarray(r, dtype=np.float64) for r in reference_pool]

    pcc_vals = [pearson_correlation(ts, r) for r in ref]
    dtw_vals = [zdtw(ts, r) for r in ref]

    cand_feats = extract_ecg_features(ts, fs)
    ref_feats  = [extract_ecg_features(r, fs) for r in ref]
    keys       = list(cand_feats.keys())

    ref_mat  = np.array([[f[k] for k in keys] for f in ref_feats])  # (n_ref, n_feats)
    cand_vec = np.array([cand_feats[k] for k in keys])              # (n_feats,)
    ref_mean = ref_mat.mean(axis=0)
    ref_std  = ref_mat.std(axis=0) + 1e-8
    z_scores = (cand_vec - ref_mean) / ref_std

    # Regularised Mahalanobis distance
    cov     = np.cov(ref_mat.T) + np.eye(len(keys)) * 1e-4
    cov_inv = np.linalg.inv(cov)
    diff    = cand_vec - ref_mean
    mahal   = float(np.sqrt(np.clip(diff @ cov_inv @ diff, 0, None)))

    return {
        'mean_pcc':             float(np.mean(pcc_vals)),
        'max_pcc':              float(np.max(pcc_vals)),
        'mean_dtw':             float(np.mean(dtw_vals)),
        'min_dtw':              float(np.min(dtw_vals)),
        'kurtosis_zscore':            float(z_scores[keys.index('kurtosis')]),
        'entropy_zscore':             float(z_scores[keys.index('sample_entropy')]),
        'subsequence_entropy_zscore': float(z_scores[keys.index('subsequence_entropy')]),
        'dominant_freq_zscore':       float(z_scores[keys.index('dominant_freq')]),
        'mahalanobis_dist':     mahal,
        'mean_abs_zscore':      float(np.mean(np.abs(z_scores))),
    }


def reidentification_score(pf):
    """
    Phase 2 — convert personalized features into a scalar.
    Higher = more likely to be the victim.

    Weights: PCC 35 % | DTW 30 % | Mahalanobis 20 % | z-score similarity 15 %
    """
    pcc_score   = (pf['max_pcc']  + pf['mean_pcc'])  / 2.0
    dtw_score   = 1.0 / (1.0 + pf['min_dtw'])
    mahal_score = 1.0 / (1.0 + pf['mahalanobis_dist'])
    zscore_sim  = 1.0 / (1.0 + pf['mean_abs_zscore'])
    return float(0.35 * pcc_score + 0.30 * dtw_score + 0.20 * mahal_score + 0.15 * zscore_sim)


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

    # --- Sample entropy (m=2, Chebyshev template matching) ---
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
    subseq_ent = ts_entropy(ts, m=min(100, len(ts) // 4), epsilon=7.0)

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

    return {
        'kurtosis':           kurt_val,
        'skewness':           skew_val,
        'std':                std_val,
        'sample_entropy':     samp_ent,
        'subsequence_entropy': subseq_ent,
        'dominant_freq':      dominant_freq,
        'spectral_entropy':   spectral_ent,
        'psd_ratio':          psd_ratio,
        'autocorr_peak_lag':  autocorr_peak_lag,
        'autocorr_peak_val':  autocorr_peak_val,
    }

def ecg_quality_filter_orphanidou(ts, fs=150, hr_min=60, hr_max=200,
                                   max_rr_s=3.0, rr_ratio_max=2.2,
                                   template_corr_threshold=0.6):
    """
    Signal quality filter based on Orphanidou et al. (2015),
    "Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
    Derivation and Applications to Wireless Monitoring", IEEE JTEHM.

    A signal passes if ALL of the following hold:
      1. Mean HR is within [hr_min, hr_max] bpm (default 40–180).
      2. Every RR interval is <= max_rr_s seconds (default 3.0 s).
      3. max(RR) / min(RR) < rr_ratio_max (default 2.2).
      4. Mean Pearson correlation of individual beats to the mean template
         >= template_corr_threshold (default 0.6).

    R-peaks are detected with Pan-Tompkins (via ecgdetectors).

    Parameters
    ----------
    ts                      : 1-D array-like ECG signal
    fs                      : sampling frequency in Hz (default 150)
    hr_min / hr_max         : HR acceptance band in bpm
    max_rr_s                : maximum allowed single RR interval in seconds
    rr_ratio_max            : maximum allowed max/min RR ratio
    template_corr_threshold : minimum mean beat-to-template correlation

    Returns
    -------
    True  if the signal passes all quality criteria
    False otherwise
    """
    ts = np.asarray(ts, dtype=np.float64)

    # --- R-peak detection ---
    try:
        detectors = Detectors(fs)
        r_peaks = np.array(detectors.pan_tompkins_detector(ts), dtype=int)
    except Exception:
        return False

    if len(r_peaks) < 2:
        return False

    rr_samples = np.diff(r_peaks)                   # RR in samples
    rr_s       = rr_samples / fs                    # RR in seconds

    if len(rr_samples) >= 6:
        return False
    # --- Criterion 1: HR range ---
    mean_rr_s = float(np.mean(rr_s))
    mean_hr   = 60.0 / mean_rr_s
    if not (hr_min <= mean_hr <= hr_max):
        return False

    # --- Criterion 2: No single RR > max_rr_s ---
    if np.any(rr_s > max_rr_s):
        return False

    # --- Criterion 3: max/min RR ratio ---
    if (rr_s.max() / (rr_s.min() + 1e-8)) >= rr_ratio_max:
        return False

    # --- Criterion 4: Template matching correlation ---
    # Extract a fixed-width window around each R-peak.
    # Window half-width = 0.3 * mean_RR (approx. QRS + T-wave).
    half_w = max(1, int(0.3 * mean_rr_s * fs))
    beats  = []
    for rp in r_peaks:
        lo, hi = rp - half_w, rp + half_w
        if lo >= 0 and hi <= len(ts):
            beat = ts[lo:hi]
            # z-normalise each beat before correlation
            std = beat.std()
            if std > 1e-8:
                beats.append((beat - beat.mean()) / std)

    if len(beats) < 2:
        return False

    # Mean template (already z-normalised beats; average then re-normalise)
    template = np.mean(beats, axis=0)
    t_std = template.std()
    if t_std < 1e-8:
        return False
    template = (template - template.mean()) / t_std

    # Pearson correlation of each beat with the template
    corrs = []
    for beat in beats:
        c = float(np.corrcoef(beat, template)[0, 1])
        corrs.append(abs(c))

    if np.max(corrs) < template_corr_threshold:
        return False

    return True


def reidentification_attack_orphanidou(candidates, reference_pool, fs=150,
                                        top_k=None,
                                        template_corr_threshold=0.9):
    """
    Re-identification attack using the Orphanidou et al. (2015) quality
    filter as Phase 1 (singling-out) instead of the soft ecg_singling_score.

    Phase 1: binary quality filter — a candidate receives phase1=1 if it
             passes all Orphanidou criteria, 0 otherwise.
    Phase 2: attribution score against the victim's reference pool
             (same as reidentification_attack; skipped if reference_pool=None).

    Parameters
    ----------
    candidates              : list of 1-D np.ndarray
    reference_pool          : list of 1-D np.ndarray (victim reference ECGs),
                              or None to run phase 1 only
    fs                      : sampling frequency in Hz
    top_k                   : return only top-k results if set
    template_corr_threshold : correlation threshold forwarded to the filter

    Returns
    -------
    List of dicts sorted by 'combined' score descending.
    Each dict contains: index, phase1, phase2, combined, + personalized features.
    """
    if reference_pool is not None:
        ref = [np.asarray(r, dtype=np.float64) for r in reference_pool]

    results = []
    for i, ts in enumerate(candidates):
        ts = np.asarray(ts, dtype=np.float64)
        ts_neg = -ts
        ts_inversed = (ts_neg - ts_neg.min()) / (ts_neg.max() - ts_neg.min())

        # Phase 1: binary quality gate — accept if either polarity passes
        # (handles ECG recordings with inverted polarity, i.e. ts or -ts)
        p1 = 1.0 if (
            ecg_quality_filter_orphanidou(ts,          fs=fs, template_corr_threshold=template_corr_threshold) or
            ecg_quality_filter_orphanidou(ts_inversed, fs=fs, template_corr_threshold=template_corr_threshold)
        ) else 0.0

        if reference_pool is not None:
            pf = extract_personalized_features(ts, ref, fs)
            p2 = reidentification_score(pf)
        else:
            pf = {}
            p2 = 1.0

        results.append({
            'index':    i,
            'phase1':   p1,
            'phase2':   p2,
            'combined': p1 * p2,
            **pf,
        })

    results.sort(key=lambda x: x['combined'], reverse=True)
    return results[:top_k] if top_k else results


def reidentification_attack(candidates, reference_pool, fs=150, top_k=None):
    """
    Two-phase re-identification attack.

    Phase 1 (singling out):  score each candidate for ECG-likeness.
    Phase 2 (attribution):   score each candidate against the victim's reference pool.
    Combined score = phase1 × phase2.

    Parameters
    ----------
    candidates:     list of 1-D np.ndarray — time series to evaluate
    reference_pool: list of 1-D np.ndarray — victim's reference ECGs (no overlap with candidates)
    fs:             sampling frequency in Hz  (default 250)
    top_k:          if set, return only the top_k results

    Returns
    -------
    List of dicts sorted by 'combined' score descending.
    Each dict contains: index, phase1, phase2, combined, + all personalized feature values.
    """
    if reference_pool is not None:
        ref = [np.asarray(r, dtype=np.float64) for r in reference_pool]
    results = []
    for i, ts in enumerate(candidates):
        ts = np.asarray(ts, dtype=np.float64)
        p1 = ecg_singling_score(ts, fs)
        if reference_pool is not None:
            pf = extract_personalized_features(ts, ref, fs)
            p2 = reidentification_score(pf)
        else:
            pf = {}
            p2 = 1
        results.append({
            'index':    i,
            'phase1':   p1,
            'phase2':   p2,
            'combined': p1 * p2,
            **pf,
        })
    results.sort(key=lambda x: x['combined'], reverse=True)
    return results[:top_k] if top_k else results


def train_reidentification_classifier(list_of_ts, fs=150, classifier=None, using_feature=True):
    """
    Train a re-identification classifier from a list of time-series groups.

    Each element ``list_of_ts[i]`` is a sequence of 1-D time series that all
    belong to identity *i*.  The function extracts ``extract_ecg_features``
    for every series, stacks them into a feature matrix, and fits a
    ``StandardScaler + classifier`` pipeline.

    Parameters
    ----------
    list_of_ts : list[list[array-like]]
        Outer list indexed by person/class (index = class label).
        Inner list contains one or more 1-D time series for that person.
    fs         : int
        Sampling frequency in Hz passed to ``extract_ecg_features``.
    classifier : sklearn estimator, optional
        Any estimator with a ``fit(X, y)`` interface.  Defaults to
        ``SVC(kernel='rbf', C=10, gamma='scale', probability=True)``.
        Examples::

            KNeighborsClassifier(n_neighbors=1)                        # 1-NN
            SVC(kernel='rbf', C=10, gamma='scale', probability=True)  # RBF-SVM (default)

    Returns
    -------
    pipeline     : sklearn.pipeline.Pipeline  (StandardScaler + classifier)
    feature_keys : list[str]
    metadata     : dict  — ``{'X': np.ndarray, 'y': np.ndarray}``

    Example
    -------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> groups = [[ts_person0_a, ts_person0_b], [ts_person1_a], ...]
    >>> pipeline, keys, meta = train_reidentification_classifier(
    ...     groups, fs=150, classifier=KNeighborsClassifier(n_neighbors=1))
    >>> feats = extract_ecg_features(new_ts, fs=150)
    >>> x_new = np.array([[feats[k] for k in keys]])
    >>> label = pipeline.predict(x_new)[0]
    """
    if classifier is None:
        print("No classifier is passed, using SVM")
        classifier = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    if args.feature:
        print("Using features extraction")
    else:
        print("Using original data")

    X_rows, y_rows = [], []
    feature_keys = None

    for class_idx, ts_group in enumerate(list_of_ts):
        for ts in ts_group:
            if using_feature:
                feats = extract_ecg_features_bis(np.asarray(ts, dtype=np.float64), fs=fs)
                if feature_keys is None:
                    feature_keys = list(feats.keys())
                X_rows.append([feats[k] for k in feature_keys])
            else:
                X_rows.append(ts)
            y_rows.append(class_idx)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int64)

    # Replace any NaN/Inf with column medians so the classifier never sees bad values
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            median = np.nanmedian(X[~bad, col]) if not np.all(bad) else 0.0
            X[bad, col] = median

    pipeline = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', classifier),
    ])
    pipeline.fit(X, y)

    return pipeline, feature_keys, {'X': X, 'y': y}


def reidentification_svm(list_of_ts, fs=150, svm_kwargs=None):
    """Backward-compatible wrapper around train_reidentification_classifier with SVC."""
    if svm_kwargs is None:
        svm_kwargs = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'probability': True}
    return train_reidentification_classifier(
        list_of_ts, fs=fs, classifier=SVC(**svm_kwargs)
    )

def reidentification_tsfresh(ts_list_train, ts_list_test, fs=150, classifier=None):
    # For a list of time series:
    dfs_train = [ts_to_tsfresh_df(ts, i) for i, ts in enumerate(ts_list_train)]
    df_train_all = pd.concat(dfs_train, ignore_index=True)

    dfs_test = [ts_to_tsfresh_df(ts, i) for i, ts in enumerate(ts_list_test)]
    df_test_all = pd.concat(dfs_test, ignore_index=True)

    X_train = extract_features(df_train_all, column_id='id', column_sort='time', column_value='value')
    impute(X_train)  # fill NaN
    y_train = np.arange(X_train.shape[0])
    # X_train_selected = select_features(X_train, y_train)
    # selected_cols = X_train_selected.columns
    # print(f"Number of features : {len(selected_cols)}")
    X_train_selected = X_train
    selected_cols = X_train.columns

    # Train and evaluate
    if classifier is None:
        print("No classifier is passed, using SVM")
        classifier = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

    pipeline = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', classifier),
    ])
    pipeline.fit(X_train_selected, y_train)

    # On test data
    X_test = extract_features(df_test_all, column_id='id', column_sort='time', column_value='value')
    impute(X_test)
    X_test_selected = X_test[selected_cols]  # apply same selection — no y_test used
    y_test = pipeline.predict(X_test_selected)

    return pipeline, selected_cols, {'X': X_test, 'y': y_test}


def evaluate_reidentification_phase1(results, true_indices, k=None):
    """Same as evaluate_reidentification but uses phase1 score for AUC (no reference pool)."""
    true_set = set(true_indices)
    k        = k or len(true_set)
    prec_top_1 = results[0]['index'] in true_indices
    topk_set = {r['index'] for r in results[:k]}
    tp       = len(topk_set & true_set)
    prec     = tp / k                 if k             > 0 else 0.0
    rec      = tp / len(true_set)     if len(true_set) > 0 else 0.0
    f1       = 2*prec*rec/(prec+rec)  if (prec + rec)  > 0 else 0.0

    y_true  = [1 if r['index'] in true_set else 0 for r in results]
    y_score = [r['phase1'] for r in results]
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')

    return {
        'precision_at_1': prec_top_1,
        'precision_at_k': prec,
        'recall_at_k':    rec,
        'f1_at_k':        f1,
        'auc_roc':        auc,
        'k':              k,
    }


def evaluate_reidentification(results, true_indices, k=None):
    """
    Evaluate a re-identification attack against ground truth labels.

    Parameters
    ----------
    results:      output of reidentification_attack (sorted list of dicts)
    true_indices: indices (into the original candidates list) that belong to the victim
    k:            evaluation cutoff — default len(true_indices)

    Returns
    -------
    dict with precision@k, recall@k, f1@k, auc_roc
    """
    

    true_set = set(true_indices)
    k        = k or len(true_set)

    topk_set = {r['index'] for r in results[:k]}
    tp       = len(topk_set & true_set)
    prec     = tp / k                 if k             > 0 else 0.0
    rec      = tp / len(true_set)     if len(true_set) > 0 else 0.0
    f1       = 2*prec*rec/(prec+rec)  if (prec + rec)  > 0 else 0.0

    y_true  = [1 if r['index'] in true_set else 0 for r in results]
    y_score = [r['combined'] for r in results]
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')

    return {
        'precision_at_k': prec,
        'recall_at_k':    rec,
        'f1_at_k':        f1,
        'auc_roc':        auc,
        'k':              k,
    }


def _load_candidates(root, victim):
    """Load all fake ECGs under root; label those from victim as positive."""
    candidates, true_indices = [], []
    person_dirs = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )
    for person_dir in person_dirs:
        person_path = os.path.join(root, person_dir)
        ecg_dirs = sorted(
            [d for d in os.listdir(person_path) if d.startswith("ecg_")],
            key=lambda x: int(x.split("_")[1]),
        )
        for ecg_dir in ecg_dirs:
            json_path = os.path.join(person_path, ecg_dir, "results.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)
            idx = len(candidates)
            candidates.append(np.array(data["data"], dtype=np.float64))
            if person_dir == victim:
                true_indices.append(idx)
    return candidates, true_indices


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-test-patient attribution breakdown")
    parser.add_argument("--mp", action="store_true", help="Use MP to reidentify")
    parser.add_argument("--dataset", type=str, default="ptbxl", help="Evaluate on dataset")
    parser.add_argument("--ipopt", action="store_true", help="Use solution after ipopt")
    parser.add_argument("--classifier", type=str, default="svm", help="The type of classifier used")
    parser.add_argument("--feature", action="store_true", help="Using feature extraction")

    args = parser.parse_args()

    if args.ipopt:
        if args.dataset == "arrhythmia":
            base_root = "src/results/ipopt/arrhythmia/"
        elif args.dataset == "ptbxl":
            base_root = "src/results/ipopt/ptbxl/"
        elif args.dataset == "arrhythmia_xl":
            base_root = "src/results/ipopt/arrhythmia_xl/"
            # base_root = "src/results/ipopt/ptbxl/ptbxl_rescale"
        elif args.dataset == "ltdb":
            base_root = "src/results/ipopt/ltdb/"
    else:
        if args.dataset == "arrhythmia":
            base_root = "test/results/ecg_arrhythmia/"
        elif args.dataset == "ptbxl":
            base_root = "src/results/baseline/ptbxl/"
        elif args.dataset == "arrhythmia_xl":
            base_root = "test/results/ecg_arrhythmia_xl/"
        elif args.dataset == "ltdb":
            base_root = "test/results/ecg_ltdb_100/"
    
    print(f"Evaluating database {args.dataset} under base root {base_root}")
    # all_persons = sorted(
    #     [d for d in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, d))]
    # )
    # print(f"Length of people in base root: {len(all_persons)}")

    # ------------------------------------------------------------------ #
    # Method 1 – Multi-person evaluation — Orphanidou quality filter      #
    # ------------------------------------------------------------------ #
    # Step 1: run the filter for every person and collect raw results
    # per_person = {}   # victim -> {'pred': set, 'true': set, 'total': int}
    # for victim in all_persons:
    #     root = os.path.join(base_root, victim)
    #     candidates, true_indices = _load_candidates(root, victim)

    #     results_orp = reidentification_attack_orphanidou(
    #         candidates, reference_pool=None, fs=128,
    #         template_corr_threshold=0.7,
    #     )
    #     pred_set  = {r['index'] for r in results_orp if r['phase1'] == 1.0}
    #     per_person[victim] = {
    #         'pred':  pred_set,
    #         'true':  set(true_indices),
    #         'total': len(candidates),mp_window_size
    #     }
    #     print(f"{victim}: {len(pred_set)} / {len(candidates)} predicted ECG-like")

    # # Step 2: for each person, remove indices that also appear in another
    # #         person's predicted_positive (ambiguous — claimed by multiple persons)
    # print("\n--- After cross-person deduplication ---")
    # for victim in all_persons:
    #     other_preds = set().union(*(
    #         per_person[v]['pred'] for v in all_persons if v != victim
    #     ))
    #     dedup_pred = per_person[victim]['pred'] - other_preds
    #     print(dedup_pred)

    #     true_set = per_person[victim]['true']
    #     total    = per_person[victim]['total']

    #     tp  = len(dedup_pred & true_set)
    #     fp  = len(dedup_pred - true_set)
    #     fn  = len(true_set - dedup_pred)
    #     tn  = total - tp - fp - fn

    #     precision = tp / (tp + fp)     if (tp + fp) > 0 else 0.0
    #     recall    = tp / (tp + fn)     if (tp + fn) > 0 else 0.0
    #     f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    #     accuracy  = (tp + tn) / total  if total > 0 else 0.0

    #     print(f"\n{victim}  (deduplicated pred={len(dedup_pred)}, removed={len(per_person[victim]['pred'])-len(dedup_pred)})")
    #     print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    #     print(f"  precision : {precision:.4f}")
    #     print(f"  recall    : {recall:.4f}")
    #     print(f"  f1        : {f1:.4f}")
    #     print(f"  accuracy  : {accuracy:.4f}")

    # ------------------------------------------------------------------ #
    # Method 2 – original soft singling-out score (commented out)        #
    # ------------------------------------------------------------------ #
    # root = "src/results/baseline/n500m100/person0"
    # victim = "person0"
    # candidates, true_indices = _load_candidates(root, victim)
    # results = reidentification_attack(candidates, reference_pool=None, fs=150)
    # results.sort(key=lambda x: x["phase1"], reverse=True)
    # metrics = evaluate_reidentification_phase1(results, true_indices)
    # print("\nRe-identification (phase 1 only, no reference pool):")
    # for key, val in metrics.items():
    #     print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # ------------------------------------------------------------------ #
    # Method 3 – Re-identification via SVM trained on real ECG reference  #
    # ------------------------------------------------------------------ #
    # Build reference pool: one list of ECGs per PTB-XL subject (attacker's data).
    # reidentification_svm expects list[list[array]] — outer index = person class.

    # Loading reference for the attacker to train the classifier
    if args.dataset == "ptbxl":
        list_patient = pd.read_csv("data/physionet.org/files/ptbxl_database.csv")["filename_lr"]
        files = sorted(list_patient[21000:21200])

    elif args.dataset=="arrhythmia" or args.dataset=="arrhythmia_xl":
        with open("data/physionet.org/files/ecg-arrhythmia/records100/RECORDS", "r") as f:
            list_patient = f.read().splitlines()
        files = [os.path.join("data/physionet.org/files/ecg-arrhythmia/records100/", file+".npy") for file in list_patient]

    elif args.dataset == "ltdb":
        list_patient = ["14046", "14134", "14149", "14157", "14172", "14184", "15814"]
        files = [os.path.join("data/physionet.org/files/ltdb/records100/", file+".npy") for file in list_patient]
    
    ref_attacker = []   # shape: (n_subjects,) each containing [ts]
    for file in files:
        if args.dataset == "ptbxl":
            record = wfdb.rdrecord(os.path.join("data/physionet.org/files/", file))
            signal = record.p_signal[:, 0].astype(np.float64)
        elif args.dataset in ("arrhythmia", "arrhythmia_xl", "ltdb"):
            signal = np.load(file)
        
        ts = normalize(signal[:500])
        if args.mp:
            mp = stumpy.stump(ts, m=100)
            ts = np.concatenate([mp[:, 0], mp[:, 1]])
            
        ref_attacker.append([ts])   # wrap in list: one sample per subject

    # Choose classifier — swap comment to switch model:
    if args.classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors=1, metric="minkowski")           # 1-NN
    elif args.classifier == "svm":
        clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)  # RBF-SVM
    elif args.classifier == "rf":
        clf = RandomForestClassifier(max_features="log2", max_depth=10)

    pipeline, feature_keys, metadata = train_reidentification_classifier(
        ref_attacker, fs=100, classifier=clf, using_feature=args.feature
    )
    print(f"Classifier: {clf.__class__.__name__}, trained on {len(ref_attacker)} subjects")

    # Build test set: flat structure — base_root/ecg_{N}/results.json
    ecg_dirs = sorted(
        [d for d in os.listdir(base_root) if d.startswith("ecg_") and d.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1]),
    )

    X_test_rows, test_labels = [], []
    for idx, ecg_dir in enumerate(ecg_dirs):
        json_path = os.path.join(base_root, ecg_dir, "results.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        if args.ipopt:
            ts = np.array(data["solutions"][0], dtype=np.float64)
        else:
            ts = np.array(data["fake_data"], dtype=np.float64)
            # ts = np.array(data["data"], dtype=np.float64)

        ts = normalize(ts)
        if args.mp:
            mp = stumpy.stump(ts, m=100)
            ts = np.concatenate([mp[:, 0], mp[:, 1]])
        if args.feature:
            feats = extract_ecg_features_bis(ts, fs=100)
            X_test_rows.append([feats[k] for k in feature_keys])
        else:
            X_test_rows.append(ts)
        # Assign ground-truth label
        if args.dataset == "arrhythmia_xl":
            test_labels.append(int(idx/5))
        elif args.dataset in ("ptbxl", "arrhythmia"):
            test_labels.append(idx)   # class index in SVM
        elif args.dataset == "ltdb":
            test_labels.append(int(idx/20))

    if not X_test_rows:
        print("No test data found under", base_root)
    else:
        X_test = np.array(X_test_rows, dtype=np.float64)
        test_labels = np.array(test_labels, dtype=np.int64)

        # Replace NaN/Inf with column medians (mirrors reidentification_svm)
        for col in range(X_test.shape[1]):
            bad = ~np.isfinite(X_test[:, col])
            if bad.any():
                X_test[bad, col] = np.nanmedian(X_test[~bad, col]) if not np.all(bad) else 0.0

        y_pred = pipeline.predict(X_test)

        correct   = (y_pred == test_labels).sum()
        incorrect = len(y_pred) - correct
        accuracy  = correct / len(y_pred)
        print(f"True  : {correct}")
        print(f"False : {incorrect}")
        print(f"Accuracy : {accuracy:.4f}")

        if args.verbose:
            print("\n--- Per-test-patient SVM attribution ---")
            for i, ptb_id in enumerate(test_labels):
                hit = "✓" if y_pred[i] == ptb_id else "✗"
                if y_pred[i] == ptb_id:
                    print(f"#{i} (class {ptb_id:3d})  → predicted #{y_pred[i]}  {hit}")

    # ------------------------------------------------------------------ #
    # Method 4 – Re-identification via tsfresh features + classifier      #
    # ------------------------------------------------------------------ #

    # Build training data: flat list of real ECGs (one per subject).
    # list_patient = pd.read_csv("data/physionet.org/files/ptbxl_database.csv")["filename_lr"]
    # files = list_patient[21000:21200]

    # ts_list_train = []
    # for file in files:
    #     record = wfdb.rdrecord(os.path.join("data/physionet.org/files/", file))
    #     signal = record.p_signal[:, 0].astype(np.float64)
    #     signal = signal[:500]
    #     r = signal.max() - signal.min()
    #     if r == 0:
    #         print(f"Constant signal: {file}")
    #     elif not np.all(np.isfinite(signal)):
    #         print(f"NaN/inf in signal: {file}")
    #     ts = normalize(signal)
    #     if args.mp:
    #         mp = stumpy.stump(ts, m=100)
    #         # ts = np.concatenate([mp[:, 0], mp[:, 1]])
    #         ts = mp[:, 0]
        
    #     ts_list_train.append(ts)

    # # Build test data: flat list of fake ECGs from base_root/ecg_{N}/results.json
    # ecg_dirs = sorted(
    #     [d for d in os.listdir(base_root) if d.startswith("ecg_")],
    #     key=lambda x: int(x.split("_")[1]),
    # )

    # ts_list_test, test_labels = [], []
    # for idx, ecg_dir in enumerate(ecg_dirs):
    #     json_path = os.path.join(base_root, ecg_dir, "results.json")
    #     if not os.path.exists(json_path):
    #         continue
    #     with open(json_path) as f:
    #         data = json.load(f)
    #     ts = np.array(data["data"], dtype=np.float64)
    #     # ts = np.array(data["solutions"][0], dtype=np.float64)
    #     ts = normalize(ts)
    #     if args.mp:
    #         mp = stumpy.stump(ts, m=100)
    #         # ts = np.concatenate([mp[:, 0], mp[:, 1]])
    #         ts = mp[:, 0]
    #     ts_list_test.append(ts)
    #     test_labels.append(idx)

    # if not ts_list_test:
    #     print("No test data found under", base_root)
    # else:
    #     clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    #     pipeline, selected_cols, result = reidentification_tsfresh(
    #         ts_list_train, ts_list_test, fs=100, classifier=clf
    #     )
    #     y_pred = result['y']
    #     test_labels = np.array(test_labels, dtype=np.int64)

    #     correct   = (y_pred == test_labels).sum()
    #     incorrect = len(y_pred) - correct
    #     accuracy  = correct / len(y_pred)
    #     print(f"Classifier: {clf.__class__.__name__} (tsfresh), "
    #           f"trained on {len(ts_list_train)} subjects, "
    #           f"{len(selected_cols)} selected features.")
    #     print(f"The selected features are : {selected_cols}")
    #     print(f"True  : {correct}")
    #     print(f"False : {incorrect}")
    #     print(f"Accuracy : {accuracy:.4f}")

    #     if args.verbose:
    #         test_ids = list(range(21000, 21200))
    #         print("\n--- Per-test-patient tsfresh attribution ---")
    #         for i, ptb_id in enumerate(test_ids):
    #             hit = "✓" if y_pred[i] == i else "✗"
    #             if y_pred[i] == i:
    #                 print(f"  PTB-XL #{ptb_id} (class {i:3d})  → predicted #{y_pred[i]}  {hit}")

    # ------------------------------------------------------------------ #
    # Method 5 – Re-identification via DTW  #
    # ------------------------------------------------------------------ #
    # Build reference pool: one list of ECGs per PTB-XL subject (attacker's data).
    # reidentification_svm expects list[list[array]] — outer index = person class.
    # list_patient = pd.read_csv("data/physionet.org/files/ptbxl_database.csv")["filename_lr"]
    # files = list_patient[21000:21200]
    # Build reference pool for Arrhythmia dataset
    # with open("data/physionet.org/files/ecg-arrhythmia/records100/RECORDS", "r") as f:
    #     list_patient = f.read().splitlines()
    # print(list_patient)
    # files = [os.path.join("data/physionet.org/files/ecg-arrhythmia/records100/", file+".npy") for file in list_patient]

    # ref_attacker = []   # shape: (n_subjects,) each containing [ts]
    # for file in files:
    #     # record = wfdb.rdrecord(os.path.join("data/physionet.org/files/", file))
    #     # signal = record.p_signal[:, 0].astype(np.float64)
    #     signal = np.load(file)
    #     ts = normalize(signal[:500])
    #     if args.mp:
    #         mp = stumpy.stump(ts, m=100)
    #         ts = np.concatenate([mp[:, 0], mp[:, 1]])
    #         ts = np.ascontiguousarray(ts, dtype=np.float64)
    #     ref_attacker.append(ts)   # wrap in list: one sample per subject

    # test_ids   = list(range(21000, 21200))
    # n_patients = len(test_ids)

    # # Build test set: flat structure — base_root/ecg_{N}/results.json
    # ecg_dirs = sorted(
    #     [d for d in os.listdir(base_root) if d.startswith("ecg_")],
    #     key=lambda x: int(x.split("_")[1]),
    # )

    # y_pred, test_labels = [], []
    # for idx, ecg_dir in enumerate(ecg_dirs):
    #     json_path = os.path.join(base_root, ecg_dir, "results.json")
    #     if not os.path.exists(json_path):
    #         continue
    #     with open(json_path) as f:
    #         data = json.load(f)
    #     ts = np.array(data["solutions"][0], dtype=np.float64)
    #     # ts = np.array(data["data"], dtype=np.float64)
    #     ts = normalize(ts)
    #     if args.mp:
    #         mp = stumpy.stump(ts, m=100)
    #         ts = np.concatenate([mp[:, 0], mp[:, 1]])
    #         ts = np.ascontiguousarray(ts, dtype=np.float64)
    #     X_test = ts
    #     # Assign ground-truth label
    #     test_labels.append(idx)   # class index in SVM
    #     y_pred.append(np.argmin(np.array([dtw.distance_fast(ref, X_test) for ref in ref_attacker])))
        
    # correct = sum(p == t for p, t in zip(y_pred, test_labels))
    # incorrect = len(y_pred) - correct
    # accuracy  = correct / len(y_pred) if y_pred else 0.0
    # print(f"True  : {correct}")
    # print(f"False : {incorrect}")
    # print(f"Accuracy : {accuracy:.4f}")
    # if args.verbose:
    #     print("\n--- Per-test-patient SVM attribution ---")
    #     for i, ptb_id in enumerate(test_ids):
    #         hit = "✓" if y_pred[i] == i else "✗"
    #         if y_pred[i] == i:
    #             print(f"  PTB-XL #{ptb_id} (class {i:3d})  → predicted #{y_pred[i]}  {hit}")