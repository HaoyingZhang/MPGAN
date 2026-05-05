"""
Parse all ecg_* folders under a results directory, load results.json from each,
compute ECG features on both real and fake time series, and save a summary CSV.

Usage:
    python test/parse_results.py --output_dir <path>
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)

from features_extraction import extract_ecg_features

RESULTS_DIR = os.path.join(root_path, "src/results/baseline/2026-03-04_10:10:37")


def parse_folder(folder_path):
    json_path = os.path.join(folder_path, "results.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    return np.array(d["data"], dtype=np.float64), np.array(d["fake_data"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy files")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    folders = sorted(
        [os.path.join(RESULTS_DIR, name) for name in os.listdir(RESULTS_DIR)
         if name.startswith("ecg_") and os.path.isdir(os.path.join(RESULTS_DIR, name))],
        key=lambda p: int(os.path.basename(p).split("_")[1])
    )

    print(f"Found {len(folders)} ecg folders in {RESULTS_DIR}")

    rows = []
    for folder in folders:
        name = os.path.basename(folder)
        result = parse_folder(folder)
        if result is None:
            print(f"  [skip] {name}: no results.json")
            continue

        real, fake = result
        print(f"  Processing {name} (len={len(real)})...", end=" ", flush=True)

        np.save(os.path.join(args.output_dir, f"{name}.npy"), real)
        np.save(os.path.join(args.output_dir, f"{name}_init.npy"), fake)

        print("done")

    df = pd.DataFrame(rows)

    # Add per-feature absolute difference columns
    feature_keys = list(next(iter([r for r in rows]), {}).keys())
    feature_keys = [k.replace("real_", "") for k in feature_keys if k.startswith("real_")]
    for key in feature_keys:
        df[f"diff_{key}"] = (df[f"real_{key}"] - df[f"fake_{key}"]).abs()

    out_path = os.path.join(RESULTS_DIR, "parsed_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print mean differences
    print("\n--- Mean absolute feature differences (real vs fake) ---")
    for key in feature_keys:
        col = f"diff_{key}"
        if col in df.columns:
            print(f"  {key:25s}: {df[col].mean():.4f}")


if __name__ == "__main__":
    main()
