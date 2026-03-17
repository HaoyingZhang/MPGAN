"""
Optuna-based tuner for reidentify.py parameters.

Tunes:
  Phase 1 (ecg_singling_score):
    - which gates to enable
    - center and scale for each gate
  Phase 2 (reidentification_score):
    - weights for pcc / dtw / mahalanobis / zscore components
  Feature extraction:
    - ts_entropy epsilon

Usage:
    python tune_reidentify.py --root src/results/baseline/n500m100 \
                               --victim person2 \
                               --n_trials 200
"""

import argparse
import json
import os
import warnings

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from reidentify import (
    extract_ecg_features,
    pearson_correlation,
    zdtw,
    ts_entropy,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candidates(root, victim):
    """
    Returns:
      candidates   : list of np.ndarray (fake ECGs from all persons)
      true_indices : list of int (indices that belong to victim)
      reference_pool: list of np.ndarray (victim's own real ECGs, if available)
    """
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
            candidates.append(np.array(data["fake_data"], dtype=np.float64))
            if person_dir == victim:
                true_indices.append(idx)

    return candidates, true_indices


def load_reference_pool(root, victim, max_ref=20):
    """Load real ECG segments for the victim from results.json 'data' field."""
    ref = []
    victim_path = os.path.join(root, victim)
    if not os.path.isdir(victim_path):
        return ref
    ecg_dirs = sorted(
        [d for d in os.listdir(victim_path) if d.startswith("ecg_")],
        key=lambda x: int(x.split("_")[1]),
    )
    for ecg_dir in ecg_dirs[:max_ref]:
        json_path = os.path.join(victim_path, ecg_dir, "results.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        if "data" in data:
            ref.append(np.array(data["data"], dtype=np.float64))
    return ref


# ---------------------------------------------------------------------------
# Cached feature extraction
# ---------------------------------------------------------------------------

def precompute_features(candidates, fs, epsilon):
    """Extract all features once; ts_entropy uses the tunable epsilon."""
    feats = []
    for ts in candidates:
        f = extract_ecg_features(ts, fs)
        # Override subsequence_entropy with the trial's epsilon
        f["subsequence_entropy"] = ts_entropy(
            ts, m=min(100, len(ts) // 4), epsilon=epsilon
        )
        feats.append(f)
    return feats


# ---------------------------------------------------------------------------
# Scoring functions built from trial parameters
# ---------------------------------------------------------------------------

def singling_score(f, params):
    """Compute phase-1 score for a single feature dict using trial params."""
    def _sig(x, center, scale):
        return 1.0 / (1.0 + np.exp(-scale * (x - center)))

    gates = []

    if params["use_kurtosis"]:
        gates.append(_sig(f["kurtosis"], params["kurt_center"], params["kurt_scale"]))

    if params["use_psd_ratio"]:
        gates.append(_sig(f["psd_ratio"], params["psd_center"], params["psd_scale"]))

    if params["use_autocorr"]:
        gates.append(_sig(f["autocorr_peak_lag"], params["ac_center"], params["ac_scale"]))

    if params["use_spectral_entropy"]:
        gates.append(1.0 - _sig(f["spectral_entropy"], params["se_center"], params["se_scale"]))

    if params["use_dominant_freq"]:
        gates.append(
            _sig(f["dominant_freq"], params["df_lo_center"], params["df_lo_scale"])
            * (1.0 - _sig(f["dominant_freq"], params["df_hi_center"], params["df_hi_scale"]))
        )

    if params["use_subseq_entropy"]:
        # Low subsequence entropy = more regular = more ECG-like → invert
        gates.append(1.0 - _sig(f["subsequence_entropy"], params["sse_center"], params["sse_scale"]))

    if not gates:
        return 0.5  # no gate selected → neutral
    return float(np.prod(gates) ** (1.0 / len(gates)))


def reidentification_score_tuned(pf, w_pcc, w_dtw, w_mahal, w_zscore):
    total = w_pcc + w_dtw + w_mahal + w_zscore + 1e-8
    pcc_score   = (pf["max_pcc"]  + pf["mean_pcc"])  / 2.0
    dtw_score   = 1.0 / (1.0 + pf["min_dtw"])
    mahal_score = 1.0 / (1.0 + pf["mahalanobis_dist"])
    zscore_sim  = 1.0 / (1.0 + pf["mean_abs_zscore"])
    return float(
        (w_pcc * pcc_score + w_dtw * dtw_score + w_mahal * mahal_score + w_zscore * zscore_sim)
        / total
    )


def compute_personalized_features_from_cache(cand_feat, ref_feats, pcc_vals, dtw_vals):
    """Compute z-scores and Mahalanobis from cached data."""
    keys = list(cand_feat.keys())
    ref_mat  = np.array([[f[k] for k in keys] for f in ref_feats])
    cand_vec = np.array([cand_feat[k] for k in keys])
    ref_mean = ref_mat.mean(axis=0)
    ref_std  = ref_mat.std(axis=0) + 1e-8
    z_scores = (cand_vec - ref_mean) / ref_std

    cov     = np.cov(ref_mat.T) + np.eye(len(keys)) * 1e-4
    cov_inv = np.linalg.inv(cov)
    diff    = cand_vec - ref_mean
    mahal   = float(np.sqrt(np.clip(diff @ cov_inv @ diff, 0, None)))

    return {
        "mean_pcc":        float(np.mean(pcc_vals)),
        "max_pcc":         float(np.max(pcc_vals)),
        "mean_dtw":        float(np.mean(dtw_vals)),
        "min_dtw":         float(np.min(dtw_vals)),
        "mahalanobis_dist": mahal,
        "mean_abs_zscore": float(np.mean(np.abs(z_scores))),
    }


def auc_score(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def make_phase1_objective(candidates, true_indices, fs, feats_cache):
    y_true = [1 if i in set(true_indices) else 0 for i in range(len(candidates))]

    def objective(trial):
        params = {
            "use_kurtosis":        trial.suggest_categorical("use_kurtosis",        [True, False]),
            "kurt_center":         trial.suggest_float("kurt_center",         0.0,  20.0),
            "kurt_scale":          trial.suggest_float("kurt_scale",          0.1,   5.0),

            "use_psd_ratio":       trial.suggest_categorical("use_psd_ratio",       [True, False]),
            "psd_center":          trial.suggest_float("psd_center",          0.3,   1.0),
            "psd_scale":           trial.suggest_float("psd_scale",           1.0,  20.0),

            "use_autocorr":        trial.suggest_categorical("use_autocorr",        [True, False]),
            "ac_center":           trial.suggest_float("ac_center",           0.2,   1.5),
            "ac_scale":            trial.suggest_float("ac_scale",            1.0,  20.0),

            "use_spectral_entropy": trial.suggest_categorical("use_spectral_entropy", [True, False]),
            "se_center":           trial.suggest_float("se_center",           0.2,   0.9),
            "se_scale":            trial.suggest_float("se_scale",            1.0,  20.0),

            "use_dominant_freq":   trial.suggest_categorical("use_dominant_freq",   [True, False]),
            "df_lo_center":        trial.suggest_float("df_lo_center",        0.2,   1.5),
            "df_lo_scale":         trial.suggest_float("df_lo_scale",         1.0,  20.0),
            "df_hi_center":        trial.suggest_float("df_hi_center",        1.5,   5.0),
            "df_hi_scale":         trial.suggest_float("df_hi_scale",         1.0,  20.0),

            "use_subseq_entropy":  trial.suggest_categorical("use_subseq_entropy",  [True, False]),
            "sse_center":          trial.suggest_float("sse_center",          0.3,   0.9),
            "sse_scale":           trial.suggest_float("sse_scale",           1.0,  20.0),
        }

        # Require at least one gate
        any_gate = any(params[k] for k in params if k.startswith("use_"))
        if not any_gate:
            return 0.5

        y_score = [singling_score(f, params) for f in feats_cache]
        return auc_score(y_true, y_score)

    return objective


def make_phase2_objective(candidates, true_indices, reference_pool, fs, feats_cache):
    if not reference_pool:
        raise ValueError("reference_pool is empty — cannot tune phase 2")

    y_true = [1 if i in set(true_indices) else 0 for i in range(len(candidates))]

    # Precompute PCC and DTW (expensive, done once)
    print("Precomputing PCC and DTW vs reference pool…")
    ref_np = [np.asarray(r, dtype=np.float64) for r in reference_pool]
    pcc_cache = []
    dtw_cache = []
    for ts in candidates:
        pcc_cache.append([pearson_correlation(ts, r) for r in ref_np])
        dtw_cache.append([zdtw(ts, r) for r in ref_np])

    ref_feats = [extract_ecg_features(r, fs) for r in ref_np]

    def objective(trial):
        w_pcc   = trial.suggest_float("w_pcc",   0.0, 1.0)
        w_dtw   = trial.suggest_float("w_dtw",   0.0, 1.0)
        w_mahal = trial.suggest_float("w_mahal", 0.0, 1.0)
        w_zsc   = trial.suggest_float("w_zscore", 0.0, 1.0)

        y_score = []
        for i, (cand_feat, pcc_vals, dtw_vals) in enumerate(
            zip(feats_cache, pcc_cache, dtw_cache)
        ):
            pf = compute_personalized_features_from_cache(
                cand_feat, ref_feats, pcc_vals, dtw_vals
            )
            y_score.append(reidentification_score_tuned(pf, w_pcc, w_dtw, w_mahal, w_zsc))

        return auc_score(y_true, y_score)

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     required=True,  help="e.g. src/results/baseline/n500m100/person2")
    parser.add_argument("--victim",   required=True,  help="e.g. person2")
    parser.add_argument("--phase",    default="1",    choices=["1", "2", "both"])
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--epsilon",  type=float, default=7.0, help="ts_entropy epsilon")
    parser.add_argument("--fs",       type=int,   default=150)
    parser.add_argument("--output",   default="optuna_reidentify.csv")
    args = parser.parse_args()

    print(f"Loading data from {args.root}, victim={args.victim}")
    candidates, true_indices = load_candidates(args.root, args.victim)
    reference_pool = load_reference_pool(args.root, args.victim)
    print(f"  {len(candidates)} candidates, {len(true_indices)} victim, {len(reference_pool)} reference")

    print(f"Precomputing features (epsilon={args.epsilon})…")
    feats_cache = precompute_features(candidates, args.fs, args.epsilon)

    sampler = optuna.samplers.TPESampler(seed=42)

    if args.phase in ("1", "both"):
        print(f"\n--- Phase 1 tuning ({args.n_trials} trials) ---")
        study1 = optuna.create_study(direction="maximize", sampler=sampler)
        study1.optimize(
            make_phase1_objective(candidates, true_indices, args.fs, feats_cache),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        print(f"Best phase-1 AUC : {study1.best_value:.4f}")
        print("Best phase-1 params:")
        for k, v in study1.best_params.items():
            print(f"  {k}: {v}")
        study1.trials_dataframe().to_csv(args.output.replace(".csv", "_phase1.csv"), index=False)

    if args.phase in ("2", "both"):
        if not reference_pool:
            print("No reference pool found — skipping phase 2 tuning.")
        else:
            print(f"\n--- Phase 2 tuning ({args.n_trials} trials) ---")
            study2 = optuna.create_study(direction="maximize", sampler=sampler)
            study2.optimize(
                make_phase2_objective(candidates, true_indices, reference_pool, args.fs, feats_cache),
                n_trials=args.n_trials,
                show_progress_bar=True,
            )
            print(f"Best phase-2 AUC : {study2.best_value:.4f}")
            print("Best phase-2 params (unnormalized weights):")
            for k, v in study2.best_params.items():
                print(f"  {k}: {v:.4f}")
            study2.trials_dataframe().to_csv(args.output.replace(".csv", "_phase2.csv"), index=False)


if __name__ == "__main__":
    main()
