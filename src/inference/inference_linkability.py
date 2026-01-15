import os, sys, argparse, random, json, time
import numpy as np
import sklearn.metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import torch
from sklearn.metrics import precision_score

# ------------------------------------------------------------
# Path handling (robust)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils_matrix_profile import MP_compute_single
from src.models.WillBeNamed import Generator
# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
N_PERSONS = 7

TRAIN_DIR = ROOT / "data/ecg/data_train_long"
TEST_DIR  = ROOT / "data/ecg/data_test_long"
MODEL_ROOT = ROOT / "src/results/baseline"

DEVICE = "cpu"

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def infer_num_ts(file_path, n, dtype=np.float32):
    bytes_per_ts = np.dtype(dtype).itemsize * n
    file_size = os.path.getsize(file_path)
    assert file_size % bytes_per_ts == 0, (
        f"File size {file_size} is not divisible by TS size {bytes_per_ts}"
    )
    return file_size // bytes_per_ts

def load_dat_file(path, n_ts, n):
    num_ts = infer_num_ts(path, n)
    ts_mm = np.memmap(path, dtype=np.float32, mode="r", shape=(num_ts, n))
    ts_mm = ts_mm[:n_ts]
    return ts_mm

def normalize_ts(ts):
    return (ts - ts.mean()) / (ts.std() + 1e-8)

# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def build_train_dataset(
    data_dir,
    n_ts,
    n
):
    X, y, person_ids = [], [], []

    for pid in range(N_PERSONS):
        path = data_dir / f"ecg_{pid}.dat"
        ts_all = load_dat_file(path, n_ts, n)

        for ts in ts_all:
            X.append(normalize_ts(ts))
            y.append(pid)
            person_ids.append(pid)

    return np.array(X), np.array(y), np.array(person_ids)

def build_test_dataset(
    data_dir,
    model_dir,
    n_ts,
    n,
    m,
    znorm_mp,
    mp_embedding,
):
    L = n - m + 1
    C = 2
    
    if mp_embedding:
        mp_dim = L
    else:
        mp_dim = C  
    G = Generator(
                n=n,
                m=m,
                mp_channels=mp_dim,
                base_channels=64,
                num_blocks=6,
                dilations=(1,2,4,8,16,32),
                use_attention=True,
                z_dim=None,
                y_dim=None,        
                use_in_proj=False,
                dropout=False
            )
    model_path = MODEL_ROOT / model_dir / "best_model.pth"
    G.load_state_dict(torch.load(model_path, map_location="cpu")
    )
    G.eval()

    X, y = [], []

    for pid in range(N_PERSONS):
        path = data_dir / f"ecg_{pid}.dat"
        ts_all = load_dat_file(path, n_ts, n)

        X_test_full = np.stack([
            MP_compute_single(
                    ts_all[i], m,
                    norm=False,
                    mpd_only=False,
                    znorm=znorm_mp,
                    embedding=mp_embedding,
                    fill_value=100
                )
                for i in range(len(ts_all))
        ])
        test_tensor = torch.stack([torch.tensor(X_test_full[i], dtype=torch.float32) 
                                for i in range(len(X_test_full))])
        with torch.no_grad():
            fake_data = G(test_tensor)

        for ts in fake_data:
            ts = normalize_ts(ts)
            X.append(ts)
            y.append(pid)

    return np.array(X), np.array(y)

def run_program(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    verbose,
):
    """
    Returns:
        dict:
            model_name -> precision_per_label (list)
    """
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision_per_label = precision_score(
            y_test,
            y_pred,
            average=None,
            zero_division=0,
        )

        results[name] = precision_per_label

        if verbose:
            for lbl, p in enumerate(precision_per_label):
                print(f"{name} – label {lbl} precision: {p:.4f}")

    return results

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_ts", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("--znorm_mp", action="store_true")
    parser.add_argument("--mp_embedding", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("-path_inversor", required=True)
    parser.add_argument("-k", type=int, required=True)  # number of runs

    args = parser.parse_args()

    rng = np.random.default_rng(42)
    seeds = rng.choice(2**12, size=args.k, replace=False)

    # model -> list of precision vectors
    values = {}

    for seed in seeds:
        print(f"Seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)

        if args.verbose:
            print("Building train set...")
        X_train, y_train, pid_train = build_train_dataset(
            TRAIN_DIR,
            args.n_ts,
            args.n,
        )

        if args.verbose:
            print("Building test set...")
        X_test, y_test = build_test_dataset(
            TEST_DIR,
            args.path_inversor,
            args.n_ts,
            args.n,
            args.m,
            args.znorm_mp,
            args.mp_embedding,
        )

        models = {
            "svm": SVC(),
            "rf": RandomForestClassifier(max_depth=20, random_state=seed),
            "knn": KNeighborsClassifier(n_neighbors=10),
        }

        run_results = run_program(
            models,
            X_train,
            y_train,
            X_test,
            y_test,
            args.verbose,
        )

        for model_name, prec_vec in run_results.items():
            values.setdefault(model_name, []).append(prec_vec)

    print("\n========== AVERAGE PRECISION PER LABEL ==========")
    for model_name, runs in values.items():
        runs = np.vstack(runs)              # shape: (k, n_labels)
        avg = runs.mean(axis=0)

        print(f"\nModel: {model_name}")
        for lbl, p in enumerate(avg):
            print(f"label {lbl} – avg precision: {p:.4f}")
    
    