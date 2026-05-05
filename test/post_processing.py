import json
import os,sys
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)  # Add root directory to path
from src.utils import normalize

def smooth_ts(ts, method="savgol", window=11, polyorder=3, cutoff=0.15, fs=1.0):
    """
    Smooth a 1-D time series.

    Parameters
    ----------
    ts        : array-like, shape (N,)
    method    : "savgol" | "butter" | "gaussian"
    window    : Savitzky-Golay window length (must be odd, >= polyorder+2)
    polyorder : Savitzky-Golay polynomial order
    cutoff    : Butterworth low-pass cutoff as a fraction of Nyquist (0 < cutoff < 1)
    fs        : sampling frequency in Hz (only used when method="butter" and cutoff is in Hz)
    """
    ts = np.asarray(ts, dtype=np.float64)
    if method == "savgol":
        w = window if window % 2 == 1 else window + 1
        return savgol_filter(ts, window_length=w, polyorder=polyorder)
    elif method == "butter":
        b, a = butter(N=4, Wn=cutoff, btype="low")
        return filtfilt(b, a, ts)
    elif method == "gaussian":
        sigma = window / 6.0
        kernel = np.exp(-0.5 * (np.arange(-window // 2, window // 2 + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        return np.convolve(ts, kernel, mode="same")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'savgol', 'butter', or 'gaussian'.")


def postprocess_ipopt_results(
    base_root,
    category="eeg",
    method="savgol",
    window=11,
    polyorder=3,
    cutoff=0.15,
    renormalize=True,
    save=True,
    out_key="smoothed",
):
    """
    Post-process IPOPT results stored under ``base_root/{category}_{N}/results.json``.

    For each result file the function:
      1. Picks the best IPOPT solution (lowest objective value) when ``best_solution=True``,
         otherwise uses ``solutions[0]``.
      2. Smooths the noisy solution with ``smooth_ts``.
      3. Optionally renormalises to [0, 1].
      4. Writes the smoothed array back to the JSON under ``out_key`` (when ``save=True``).

    Parameters
    ----------
    base_root     : str   — root directory (contains ``eeg_0``, ``eeg_1``, … sub-dirs)
    category      : str   — prefix of sub-directories (e.g. "eeg", "ecg")
    method        : str   — smoothing method passed to ``smooth_ts``
    window        : int   — smoothing window length
    polyorder     : int   — Savitzky-Golay polynomial order
    cutoff        : float — Butterworth cutoff (fraction of Nyquist)
    best_solution : bool  — pick the solution with the lowest objective value
    renormalize   : bool  — normalise the smoothed signal to [0, 1]
    save          : bool  — write the result back into results.json
    out_key       : str   — key under which the smoothed array is stored in the JSON

    Returns
    -------
    results : list[dict]
        One entry per processed file with keys ``dir``, ``raw``, ``smoothed``,
        ``objective_idx`` (index of the chosen solution), ``objective_value``.
    """
    dirs = sorted(
        [d for d in os.listdir(base_root) if d.startswith(f"{category}_") and d.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1]),
    )

    processed = []
    for d in dirs:
        json_path = os.path.join(base_root, d, "results.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        solutions = data.get("solutions", [])

        raw = np.array(solutions[0], dtype=np.float64)
        smoothed = smooth_ts(raw, method=method, window=window, polyorder=polyorder, cutoff=cutoff)

        if renormalize:
            smoothed = normalize(smoothed)

        processed.append({
            "dir": d,
            "raw": raw,
            "smoothed": smoothed
        })

        if save:
            data[out_key] = smoothed.tolist()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

    print(f"Processed {len(processed)} files from {base_root}")
    return processed


def spectrum_match_ts(solution, reference, alpha=1.0, match_moments=True):
    """
    Frequency-domain amplitude matching: keep the phases of `solution` but
    replace (or blend) its amplitude spectrum with that of `reference`.

    This directly preserves all power-spectral features of the reference
    (band powers, spectral entropy, AR coefficients, aperiodic exponent)
    while retaining the adversarial temporal structure of the solution.

    Parameters
    ----------
    solution      : array-like  — IPOPT reconstructed signal
    reference     : array-like  — original time series
    alpha         : float in [0,1]  — 1.0 = full amplitude substitution,
                    <1.0 = linear blend between solution and reference amplitudes
    match_moments : bool  — after spectrum matching, rescale mean/std to match
                    the reference (fixes kurtosis baseline drift)

    Returns
    -------
    np.ndarray, same length as solution
    """
    sol = np.asarray(solution, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    F_sol = np.fft.rfft(sol)
    F_ref = np.fft.rfft(ref)

    amp_new = (1.0 - alpha) * np.abs(F_sol) + alpha * np.abs(F_ref)
    F_new   = amp_new * np.exp(1j * np.angle(F_sol))

    result = np.fft.irfft(F_new, n=len(sol))

    if match_moments:
        result = (result - result.mean()) / (result.std() + 1e-8)
        result = result * (ref.std() + 1e-8) + ref.mean()

    return result


def postprocess_spectrum_match(
    base_root,
    category="eeg",
    alpha=1.0,
    match_moments=True,
    renormalize=True,
    save=True,
    out_key="spectrum_matched",
):
    """
    Apply frequency-domain amplitude matching to every IPOPT result under
    ``base_root/{category}_{N}/results.json``.

    For each file the function:
      1. Loads ``solutions[0]`` (the IPOPT output) and ``time_series`` (the original).
      2. Runs ``spectrum_match_ts`` to align the amplitude spectrum of the solution
         to the original while keeping the solution's phases.
      3. Optionally renormalises the result to [0, 1].
      4. Writes the array back under ``out_key`` when ``save=True``.

    Parameters
    ----------
    base_root     : str
    category      : str   — sub-directory prefix ("eeg", "ecg", …)
    alpha         : float — blending factor (1.0 = full substitution)
    match_moments : bool  — also match mean/std after spectrum correction
    renormalize   : bool  — normalise output to [0, 1]
    save          : bool  — write result back into results.json
    out_key       : str   — JSON key for the processed result

    Returns
    -------
    list[dict] with keys ``dir``, ``solution``, ``reference``, ``matched``
    """
    dirs = sorted(
        [d for d in os.listdir(base_root)
         if d.startswith(f"{category}_") and d.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1]),
    )

    processed = []
    for d in dirs:
        json_path = os.path.join(base_root, d, "results.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        solution  = np.array(data["solutions"][0], dtype=np.float64)
        reference = np.array(data["time_series"],  dtype=np.float64)

        matched = spectrum_match_ts(solution, reference, alpha=alpha, match_moments=match_moments)

        if renormalize:
            matched = normalize(matched)

        processed.append({
            "dir":       d,
            "solution":  solution,
            "reference": reference,
            "matched":   matched,
        })

        if save:
            data[out_key] = matched.tolist()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

    print(f"Spectrum-matched {len(processed)} files from {base_root}")
    return processed


def spectrum_selfref_ts(solution, smooth_window=21, polyorder=3, renorm_1f=True):
    """
    Self-referencing spectral smoothing: no reference signal required.

    The attacker only uses `solution` itself — it smooths the amplitude
    spectrum of the IPOPT output and optionally re-applies a 1/f envelope,
    which regularises the signal toward natural EEG structure without
    requiring knowledge of the original recording.

    Steps
    -----
    1. Compute rfft of the solution.
    2. Smooth the amplitude spectrum with a Savitzky-Golay filter
       (removes sharp spectral spikes caused by the IPOPT optimiser).
    3. If renorm_1f=True, fit a log-linear (1/f) trend to the smoothed
       amplitudes and divide it out before smoothing, then multiply back in.
       This preserves the aperiodic slope while cleaning per-bin artefacts.
    4. Reconstruct via irfft (phases are unchanged).

    Parameters
    ----------
    solution      : array-like  — IPOPT reconstructed signal (no reference needed)
    smooth_window : int         — Savitzky-Golay window over frequency bins (odd)
    polyorder     : int         — Savitzky-Golay polynomial order
    renorm_1f     : bool        — fit and preserve the 1/f aperiodic envelope

    Returns
    -------
    np.ndarray, same length as solution
    """
    sol = np.asarray(solution, dtype=np.float64)
    F   = np.fft.rfft(sol)
    amp = np.abs(F)
    phase = np.angle(F)

    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1

    if renorm_1f:
        # Fit 1/f envelope in log-log space (skip DC bin)
        freqs_idx = np.arange(1, len(amp))
        log_f = np.log(freqs_idx.astype(np.float64))
        log_a = np.log(amp[1:] + 1e-12)
        slope, intercept = np.polyfit(log_f, log_a, 1)
        envelope = np.exp(intercept + slope * log_f)
        # Whiten, smooth, re-colour
        whitened = amp[1:] / (envelope + 1e-12)
        whitened_smooth = savgol_filter(whitened, window_length=w, polyorder=polyorder)
        whitened_smooth = np.clip(whitened_smooth, 0, None)
        amp_new = np.concatenate([[amp[0]], whitened_smooth * envelope])
    else:
        amp_smooth = savgol_filter(amp, window_length=w, polyorder=polyorder)
        amp_new = np.clip(amp_smooth, 0, None)

    result = np.fft.irfft(amp_new * np.exp(1j * phase), n=len(sol))
    return result


def postprocess_selfref(
    base_root,
    category="eeg",
    smooth_window=21,
    polyorder=3,
    renorm_1f=True,
    renormalize=True,
    save=True,
    out_key="selfref_matched",
):
    """
    Apply self-referencing spectral smoothing to every IPOPT result under
    ``base_root/{category}_{N}/results.json``.

    Unlike ``postprocess_spectrum_match``, this requires no reference signal —
    the attacker only needs the IPOPT solution itself.

    Parameters
    ----------
    base_root     : str
    category      : str   — sub-directory prefix ("eeg", "ecg", …)
    smooth_window : int   — Savitzky-Golay window over frequency bins
    polyorder     : int   — Savitzky-Golay polynomial order
    renorm_1f     : bool  — preserve 1/f aperiodic envelope
    renormalize   : bool  — normalise output to [0, 1]
    save          : bool  — write result back into results.json
    out_key       : str   — JSON key for the processed result

    Returns
    -------
    list[dict] with keys ``dir``, ``solution``, ``selfref``
    """
    dirs = sorted(
        [d for d in os.listdir(base_root)
         if d.startswith(f"{category}_") and d.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1]),
    )

    processed = []
    for d in dirs:
        json_path = os.path.join(base_root, d, "results.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        solution = np.array(data["solutions"][0], dtype=np.float64)
        selfref  = spectrum_selfref_ts(solution, smooth_window=smooth_window,
                                       polyorder=polyorder, renorm_1f=renorm_1f)

        if renormalize:
            selfref = normalize(selfref)

        processed.append({"dir": d, "solution": solution, "selfref": selfref})

        if save:
            data[out_key] = selfref.tolist()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

    print(f"Self-ref processed {len(processed)} files from {base_root}")
    return processed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_root", help="Directory with eeg_*/results.json files")
    parser.add_argument("--category",      default="eeg", help="Sub-directory prefix")
    parser.add_argument("--mode",          default="smooth", choices=["smooth", "spectrum", "selfref"],
                        help="'smooth': time-domain  |  'spectrum': amplitude matching (needs reference)  |  'selfref': self-referencing spectral smoothing")
    # smooth options
    parser.add_argument("--method",        default="savgol", choices=["savgol", "butter", "gaussian"])
    parser.add_argument("--window",        type=int,   default=11)
    parser.add_argument("--polyorder",     type=int,   default=3)
    parser.add_argument("--cutoff",        type=float, default=0.15)
    # spectrum-match options
    parser.add_argument("--alpha",         type=float, default=1.0,
                        help="Amplitude blend factor (1.0 = full substitution)")
    parser.add_argument("--no-moments",    action="store_true", help="Skip mean/std matching")
    # selfref options
    parser.add_argument("--smooth-window", type=int,   default=21, help="Savitzky-Golay window over freq bins")
    parser.add_argument("--no-1f",         action="store_true",    help="Skip 1/f envelope preservation")
    # shared
    parser.add_argument("--no-save",       action="store_true", help="Dry run — do not write back to JSON")
    parser.add_argument("--out-key",       default=None, help="JSON key for the result (auto if omitted)")
    args = parser.parse_args()

    if args.mode == "selfref":
        out_key = args.out_key or "selfref_matched"
        results = postprocess_selfref(
            base_root=args.base_root,
            category=args.category,
            smooth_window=args.smooth_window,
            polyorder=args.polyorder,
            renorm_1f=not args.no_1f,
            save=not args.no_save,
            out_key=out_key,
        )
        print(f"Wrote '{out_key}' into {len(results)} files.")

    elif args.mode == "spectrum":
        out_key = args.out_key or "spectrum_matched"
        results = postprocess_spectrum_match(
            base_root=args.base_root,
            category=args.category,
            alpha=args.alpha,
            match_moments=not args.no_moments,
            save=not args.no_save,
            out_key=out_key,
        )
        from features_extraction import extract_eeg_features
        errs_sol, errs_mat = [], []
        for r in results:
            f_ref = extract_eeg_features(r["reference"], fs=500)
            f_sol = extract_eeg_features(r["solution"],  fs=500)
            f_mat = extract_eeg_features(r["matched"],   fs=500)
            errs_sol.append(np.mean([abs(f_sol[k]-f_ref[k])/(abs(f_ref[k])+1e-8) for k in f_ref]))
            errs_mat.append(np.mean([abs(f_mat[k]-f_ref[k])/(abs(f_ref[k])+1e-8) for k in f_ref]))
        print(f"Mean feature rel-error  solution : {np.mean(errs_sol):.4f}")
        print(f"Mean feature rel-error  matched  : {np.mean(errs_mat):.4f}")
    else:
        out_key = args.out_key or "smoothed"
        results = postprocess_ipopt_results(
            base_root=args.base_root,
            category=args.category,
            method=args.method,
            window=args.window,
            polyorder=args.polyorder,
            cutoff=args.cutoff,
            save=not args.no_save,
            out_key=out_key,
        )
        diffs = [np.std(r["smoothed"] - r["raw"]) for r in results]
        print(f"Mean smoothing delta : {np.mean(diffs):.4f}")
        print(f"Max  smoothing delta : {np.max(diffs):.4f}")
