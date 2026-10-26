"""
Re-identification rate and reconstruction PCC of the IPOPT reconstructions from the noisy MP.

attack_matrix_profile/src/rmpi/exhaustive_rmpi_noisy_mp.py stores, in every <category>_i/results.json
of a robustness folder, the series reconstructed from the noisy MP under the key "reconstruct"
(warm-started from the DeepMP output "fake_data"). For every eps_mpi in --eps_mpi this computes
  - the PCC between the original series ("data") and "reconstruct", per sample (and, for
    comparison, the PCC of "fake_data"),
  - the re-identification rate of the attack on "reconstruct" (same attack as test/eval_robustness.py),
and appends them to --results as {dataset: {eps: {metric: [run 1, run 2, ...]}}}.

Usage (from the repository root):
    python3 test/eval_reconstruct.py
    python3 test/eval_reconstruct.py --dataset ptbxl --eps_mpi 10000 100000 500000 --eps_mpd 100000
"""
import argparse, json, os, re, sys
import numpy as np

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)  # Add root directory to path

from src.utils import pearson_correlation
from reidentify import reidentification_attack

RESULTS_ROOT = {"ecg": "src/results/baseline/ptbxl/", "eeg": "src/results/baseline/eeg_160hz/"}


def pcc_from_folder(folder, category, key):
    """PCC between "data" and `key` for every <category>_i/results.json of folder, ordered by i."""
    names = sorted((name for name in os.listdir(folder) if re.fullmatch(rf"{category}_\d+", name)),
                   key=lambda name: int(name.split("_")[1]))
    pccs = []
    for name in names:
        with open(os.path.join(folder, name, "results.json")) as f:
            d = json.load(f)
        pccs.append(float(pearson_correlation(np.asarray(d["data"], dtype=np.float64),
                                              np.asarray(d[key], dtype=np.float64))))
    return pccs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl")
    ap.add_argument("--category", default="ecg", choices=list(RESULTS_ROOT))
    ap.add_argument("--eps_mpi", type=float, nargs="+", default=[10000.0, 100000.0, 500000.0])
    ap.add_argument("--eps_mpd", type=float, default=100000.0, help="fixed MPD budget of the folders")
    ap.add_argument("--key", default="reconstruct", help="results.json key of the attacked series")
    ap.add_argument("--results", default=None, help="json the runs are appended to")
    args = ap.parse_args()

    results_path = args.results or os.path.join(RESULTS_ROOT[args.category],
                                                f"eval_{args.key}_mpd{args.eps_mpd}_results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    runs_ds = all_results.setdefault(args.dataset, {})

    for eps_mpi in args.eps_mpi:
        folder = os.path.join(RESULTS_ROOT[args.category], args.dataset, "both", f"mpi{eps_mpi}_mpd{args.eps_mpd}")
        pcc = pcc_from_folder(folder, args.category, args.key)
        pcc_fake = pcc_from_folder(folder, args.category, "fake_data")
        rir = float(reidentification_attack(folder, ipopt=False, dataset=args.dataset, category=args.category,
                                            use_mp=False, classifier="svm", use_feature=True,
                                            use_original=False, distance=False, ts_key=args.key))
        run = {"reconstruction_pcc": pcc, "fake_pcc": pcc_fake, "reidentification_rate": rir}
        runs = runs_ds.setdefault(str(eps_mpi), {})
        for name, v in run.items():
            runs[name] = runs.get(name, []) + [v]
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[{args.dataset}] eps_mpi={eps_mpi} eps_mpd={args.eps_mpd} ({len(pcc)} samples): "
              f"PCC {args.key}={np.nanmean(pcc):.4f} (fake_data={np.nanmean(pcc_fake):.4f}) "
              f"RIR {args.key}={rir:.4f}")
    print(f"Appended to {results_path}")


if __name__ == "__main__":
    main()
