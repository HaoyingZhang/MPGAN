import os, sys, argparse, random, json, time
import numpy as np
import sklearn.metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import torch

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
AGES = [46, 71, 47, 88, 75, 71, 58]  # ecg_0 ... ecg_6
N_PERSONS = len(AGES)

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

def age_to_label(age):
    return int(age < 60)

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
            y.append(age_to_label(AGES[pid]))
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

    X, y, person_ids = [], [], []

    for pid in range(N_PERSONS):
        path = data_dir / f"ecg_{pid}.dat"
        ts_all = load_dat_file(path, n_ts, n)

        X_test_full = np.stack([
            MP_compute_single(
                    ts_all[i], m,
                    norm=False,
                    mpd_only=False,
                    znorm=znorm_mp,
                    embedding=mp_embedding
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
            y.append(age_to_label(AGES[pid]))
            person_ids.append(pid)

    return np.array(X), np.array(y), np.array(person_ids)

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

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print("Building train set...")
    X_train, y_train, pid_train = build_train_dataset(
        TRAIN_DIR,
        args.n_ts,
        args.n
    )

    if args.verbose:
        print("Building test set...")
    X_test, y_test, pid_test = build_test_dataset(
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
        "rf": RandomForestClassifier(max_depth=20, random_state=args.seed),
        "knn": KNeighborsClassifier(n_neighbors=10),
    }

    results = {}

    for name, model in models.items():
        if args.verbose:
            print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {}

        for pid in range(N_PERSONS):
            mask = pid_test == pid
            cm = sklearn.metrics.confusion_matrix(
                y_test[mask],
                y_pred[mask],
                labels=[0, 1],
            )
            results[name][f"person_{pid}"] = cm.tolist()

            if args.verbose:
                print(f"{name} – person {pid} confusion matrix:\n{cm}")

    # Optional save
    out_dir = ROOT / "src/results/attribute_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "age_confusion_matrices.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to {out_path}")