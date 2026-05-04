import numpy as np
import os, json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.stats import gaussian_kde
from dtaidistance.dtw import distance as dtw_distance
from scipy.spatial.distance import euclidean
import wfdb
from src.utils_matrix_profile import normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from features_extraction import extract_ecg_features, extract_eeg_features, ts_to_tsfresh_df, extract_eeg_features_bis
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import stumpy
import mne
from src.utils import pearson_correlation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from loader import ptbxl_loader, arrhythmia_loader, ltdb_loader, tdbrain_loader

## GLOBAL VARIABLES

DATA_LOADERS = {"ptbxl": ptbxl_loader, "arrhythmia_xl": arrhythmia_loader, "ltdb": ltdb_loader, "tdbrain": tdbrain_loader}
LIST_PEOPLE =  {"ptbxl": np.arange(21000,21200), "arrhythmia_xl": np.arange(48), "ltdb": np.arange(7), "tdbrain": np.arange(1000,1200)}
SOURCE_HZ = {"ptbxl": 100, "arrhythmia_xl": 100, "ltdb": 100, "tdbrain": 500}
SESSION_EEG = "EC"

class _TransformerNet(nn.Module):
    def __init__(self, seq_len, n_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.input_proj(x.unsqueeze(-1)) + self.pos_emb(positions)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


class _CNNNet(nn.Module):
    def __init__(self, n_classes, channels=(32, 64, 128), kernel_size=7, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))   # (B, C, L)
        x = x.mean(dim=-1)              # global average pool
        return self.head(self.drop(x))


class TimeSeriesCNNClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible 1-D CNN classifier for raw time series."""

    def __init__(self, channels=(32, 64, 128), kernel_size=7, dropout=0.3,
                 epochs=50, batch_size=16, lr=1e-3, device=None):
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.model_ = _CNNNet(n_classes, self.channels, self.kernel_size, self.dropout).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model_(xb), yb).backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32).to(self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()


class TimeSeriesTransformerClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible Transformer encoder classifier for raw time series."""

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.1, epochs=50, batch_size=16, lr=1e-3, device=None):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        seq_len = X.shape[1]
        self.model_ = _TransformerNet(seq_len, n_classes, self.d_model, self.nhead,
                                      self.num_layers, self.dim_feedforward, self.dropout).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for i in range(self.epochs):
            print(f"Epoch {i+1}/{self.epochs}")
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model_(xb), yb).backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32).to(self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def plot_feature_distributions(X, feature_keys, labels=None, title="Test-set feature distributions", out="feature_distributions.png"):
    """
    KDE grid: one panel per feature. Each person's density is drawn as a thin
    coloured line; the thick blue fill is the population-level KDE.
    """
    n_feat = len(feature_keys)
    ncols = 5
    nrows = math.ceil(n_feat / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.6),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    unique_labels = np.unique(labels) if labels is not None else []
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    for fi, (ax, feat) in enumerate(zip(axes, feature_keys)):
        col = X[:, fi]
        finite = col[np.isfinite(col)]
        if len(finite) < 2:
            ax.set_title(feat, fontsize=7)
            continue

        lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
        pad = max((hi - lo) * 0.15, 1e-6)
        x_grid = np.linspace(lo - pad, hi + pad, 300)

        # Per-person thin KDE
        for li, lbl in enumerate(unique_labels):
            vals = col[labels == lbl]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            try:
                ax.plot(x_grid, gaussian_kde(vals)(x_grid),
                        color=cmap(li), alpha=0.3, linewidth=0.8)
            except Exception:
                pass

        # Population KDE
        try:
            kde_all = gaussian_kde(finite)
            ax.fill_between(x_grid, kde_all(x_grid), alpha=0.25, color="steelblue")
            ax.plot(x_grid, kde_all(x_grid), color="steelblue", linewidth=1.5)
        except Exception:
            pass

        ax.set_title(feat, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Feature distribution plot saved → {out}")
    plt.close(fig)


def plot_individual_separation(X, feature_keys, labels, title="Individual separation", out_prefix="separation"):
    """
    Three plots for per-subject discriminability:
      1. Fisher score bar chart — ranks features by inter/intra-class variance ratio
      2. Ridgeline plot — per-subject KDEs stacked for the top-10 Fisher features
      3. PCA scatter — all samples projected to 2-D and coloured by subject
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_subjects = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", max(n_subjects, 1))

    # Fisher scores
    overall_mean = np.nanmean(X, axis=0)
    inter = np.zeros(X.shape[1])
    intra = np.zeros(X.shape[1])
    for lbl in unique_labels:
        mask = labels == lbl
        n = mask.sum()
        class_mean = np.nanmean(X[mask], axis=0)
        inter += n * (class_mean - overall_mean) ** 2
        intra += np.nanvar(X[mask], axis=0) * n
    intra = np.where(intra == 0, 1e-12, intra)
    fisher = inter / intra
    order = np.argsort(fisher)[::-1]

    # 1. Fisher score bar chart (top 20)
    top_n = min(20, len(feature_keys))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(range(top_n), fisher[order[:top_n]], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_keys[i] for i in order[:top_n]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fisher score")
    ax.set_title(f"{title} — top-{top_n} features by Fisher score")
    fig.savefig(f"{out_prefix}_fisher.png", dpi=150, bbox_inches="tight")
    print(f"Fisher score plot saved → {out_prefix}_fisher.png")
    plt.close(fig)

    # 2. Ridgeline: top-10 features, up to 30 subjects for readability
    top_k = min(10, len(feature_keys))
    display_labels = unique_labels[:30]  # cap at 30 subjects
    ncols = 5
    nrows = math.ceil(top_k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * max(len(display_labels) * 0.25 + 1, 3)),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for fi in range(top_k):
        ax = axes[fi]
        feat_idx = order[fi]
        col = X[:, feat_idx]
        finite_all = col[np.isfinite(col)]
        if len(finite_all) < 2:
            continue
        lo, hi = np.percentile(finite_all, 1), np.percentile(finite_all, 99)
        pad = max((hi - lo) * 0.15, 1e-6)
        x_grid = np.linspace(lo - pad, hi + pad, 200)

        for li, lbl in enumerate(display_labels):
            vals = col[labels == lbl]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            try:
                density = gaussian_kde(vals)(x_grid)
                offset = li * density.max() * 1.8
                ax.fill_between(x_grid, offset, offset + density, alpha=0.45, color=cmap(li))
                ax.plot(x_grid, offset + density, color=cmap(li), linewidth=0.7)
            except Exception:
                pass

        ax.set_title(f"{feature_keys[feat_idx]}\n(F={fisher[feat_idx]:.2f})", fontsize=7, pad=2)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)

    for ax in axes[top_k:]:
        ax.set_visible(False)

    fig.suptitle(f"{title} — ridgeline (top-{top_k})", fontsize=10)
    fig.savefig(f"{out_prefix}_ridgeline.png", dpi=150, bbox_inches="tight")
    print(f"Ridgeline plot saved → {out_prefix}_ridgeline.png")
    plt.close(fig)

    # 3. PCA scatter coloured by subject
    X_clean = np.where(np.isfinite(X), X, 0.0)
    coords = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_clean))

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for li, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], color=cmap(li), s=30, alpha=0.7, label=str(lbl))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"{title} — PCA")
    if n_subjects <= 20:
        ax.legend(title="Subject", fontsize=6, ncol=2, loc="best")
    fig.savefig(f"{out_prefix}_pca.png", dpi=150, bbox_inches="tight")
    print(f"PCA scatter saved → {out_prefix}_pca.png")
    plt.close(fig)


def plot_feature_preservation(ref_vecs, test_vecs, subject_ids, feature_keys=None, out_prefix="feature_preservation"):
    """
    For each subject, compute the Euclidean distance between their reference
    feature vector and their reconstructed test feature vector, then plot a
    bar chart sorted from most preserved (smallest distance) to least and
    print the ranking.

    Parameters
    ----------
    ref_vecs    : list[np.ndarray]  — one reference vector per subject (scaled)
    test_vecs   : list[np.ndarray]  — one test vector per subject (same order, scaled)
    subject_ids : list[int]         — subject label for each entry
    feature_keys: list[str] | None  — feature names (for per-feature breakdown)
    out_prefix  : str               — prefix for saved PNG files
    """
    n = len(ref_vecs)
    dists = np.array([euclidean(ref_vecs[i], test_vecs[i]) for i in range(n)])
    order = np.argsort(dists)

    sorted_ids   = [subject_ids[i] for i in order]
    sorted_dists = dists[order]

    print("\n--- Feature preservation ranking (nearest → least preserved) ---")
    for rank, (sid, d) in enumerate(zip(sorted_ids, sorted_dists)):
        print(f"  #{rank+1:3d}  subject {sid:3d}  dist={d:.4f}")

    # Bar chart of sorted distances
    fig, ax = plt.subplots(figsize=(max(8, n * 0.15), 4), constrained_layout=True)
    ax.bar(range(n), sorted_dists, color="steelblue")
    ax.set_xlabel("Subject rank (most → least preserved)", fontsize=11)
    ax.set_ylabel("Feature distance (Euclidean, scaled)", fontsize=11)
    ax.set_title("Feature preservation: reference vs reconstructed", fontsize=12)
    if n <= 50:
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(s) for s in sorted_ids], rotation=90, fontsize=7)
    out_path = f"{out_prefix}_distances.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Feature preservation plot saved → {out_path}")
    plt.close(fig)

    # Per-feature breakdown: mean relative error per feature.
    # Using |ref − test| / (|ref| + ε) so that features with larger absolute
    # values do not artificially dominate the ranking.
    if feature_keys is not None:
        ref_mat  = np.array(ref_vecs,  dtype=np.float64)
        test_mat = np.array(test_vecs, dtype=np.float64)
        eps = 1e-6
        rel_err = (np.abs(ref_mat - test_mat) / (np.abs(ref_mat) + eps)).mean(axis=0)
        feat_order = np.argsort(rel_err)

        fig2, ax2 = plt.subplots(figsize=(max(8, len(feature_keys) * 0.3), 4), constrained_layout=True)
        ax2.bar(range(len(feature_keys)), rel_err[feat_order] * 100, color="coral")
        ax2.set_xticks(range(len(feature_keys)))
        ax2.set_xticklabels([feature_keys[i] for i in feat_order], rotation=90, fontsize=7)
        ax2.set_xlabel("Feature (most → least preserved)", fontsize=11)
        ax2.set_ylabel("Mean relative error (%)", fontsize=11)
        ax2.set_title("Per-feature preservation (relative error)", fontsize=12)
        ax2.set_yscale("log")
        out_path2 = f"{out_prefix}_per_feature.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        print(f"Per-feature preservation plot saved → {out_path2}")
        plt.close(fig2)


def train_reidentification_classifier(list_of_ts, fs=150, classifier=None, using_feature=True, category="ecg", robust_features=False, n_features=None):
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
    if using_feature:
        print("Using features extraction")
    else:
        print("Using original data")

    X_rows, y_rows = [], []
    feature_keys = None

    for class_idx, ts_group in enumerate(list_of_ts):
        for ts in ts_group:
            if using_feature:
                if category == "ecg":
                    feats = extract_ecg_features(np.asarray(ts, dtype=np.float64), fs=fs)
                elif category == "eeg":
                    feats = extract_eeg_features(np.asarray(ts, dtype=np.float64), fs=fs, robust=robust_features)
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

    if using_feature and n_features is not None:
        k = min(n_features, X.shape[1])
        selector = SelectKBest(f_classif, k=k)
        X = selector.fit_transform(X, y)
        mask = selector.get_support()
        feature_keys = [fk for fk, m in zip(feature_keys, mask) if m]

    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', classifier)])
    print(X.shape, y.shape)
    pipeline.fit(X, y)

    return pipeline, feature_keys, {'X': X, 'y': y}

def reidentification_attack(base_root, n = 500, 
                            ipopt=False, dataset="ptbxl",
                            category="ecg", use_mp=False, 
                            classifier="svm", use_feature=True, 
                            use_original=False, distance=False, 
                            metric="dtw", verbose=True, 
                            resample_hz=None, plot=False, 
                            robust_features=False, n_features=None):
    print(f"Evaluating database {dataset} under base root {base_root}")
    
    REFERENCE_INDICES = np.concatenate((np.arange(1000, 5000, step=n), np.arange(6000, 8000, step=n))) if dataset == "tdbrain" else [0]

    frequency = SOURCE_HZ[dataset] if resample_hz is None else resample_hz
    if dataset == "tdbrain":
        all_ts = DATA_LOADERS[dataset](LIST_PEOPLE[dataset], dest_hz=frequency, session=SESSION_EEG)
    else:
        all_ts = DATA_LOADERS[dataset](LIST_PEOPLE[dataset], dest_hz=frequency)
    
    # Build train set: Loade attacker reference TS
    ref_attacker = []
    ids = []
    labels = []

    for idx, long_ts_person in enumerate(all_ts):
        ref_ts_person = []
        for ind in REFERENCE_INDICES:
            ts = normalize(long_ts_person[ind:ind+n])
            if use_mp:
                mp = stumpy.stump(ts, m=100)
                ts = np.concatenate([mp[:, 0], mp[:, 1]])
            ref_ts_person.append(ts)
        ref_attacker.append(ref_ts_person)
        ids.append(idx)

    print(f"{len(ref_attacker)} People loaded in the training set")

    # Build test set: flat structure — base_root/ecg_{N}/results.json
    ts_raw_test = []

    if use_original:
        for label_test in ids:
            ts = np.array(all_ts[label_test][500:500+n], dtype=np.float64)
            if use_mp:
                mp = stumpy.stump(np.array(ts, dtype=np.float64), m=100)
                ts = np.concatenate([mp[:, 0], mp[:, 1]])
            ts = normalize(ts)
            ts_raw_test.append(ts)
    else:
        test_dirs = sorted(
            [os.path.join(base_root, d) for d in os.listdir(base_root)
             if d.startswith(f"{category}_") and d.split("_")[1].isdigit()],
            key=lambda x: int(os.path.basename(x).split("_")[1]),
        )
        ids = np.arange(len(test_dirs))
    
        for ecg_dir in test_dirs:
            json_path = os.path.join(ecg_dir, "results.json")
            if not os.path.exists(json_path):
                print(f"FATAL: {ecg_dir} not found !")
                return 
            with open(json_path) as f:
                data = json.load(f)
            if ipopt:
                if dataset == "tdbrain":
                    ts = np.array(data["smoothed"], dtype=np.float64)
                    # ts = np.array(data["solutions"][0], dtype=np.float64)
                else:
                    ts = np.array(data["solutions"][0], dtype=np.float64)
            else:
                ts = np.array(data["fake_data"], dtype=np.float64)

            ts = normalize(ts)
            if use_mp:
                mp = stumpy.stump(np.array(ts, dtype=np.float64), m=100)
                ts = np.concatenate([mp[:, 0], mp[:, 1]])
            ts_raw_test.append(ts)
        

    # Assign ground-truth label
    if dataset in ("arrhythmia_xl", "arrhythmia"):
        labels = [int(id/5) for id in ids]
    elif dataset in ("ptbxl", "tdbrain"):
        labels = ids  
    elif dataset == "ltdb":
        labels = [int(id/20) for id in ids]
    
    # Training the classifier
    accuracy = 0

    # Using classifier
    if not distance:
    # Choose classifier — swap comment to switch model:
        if classifier == "knn":
            clf = KNeighborsClassifier(n_neighbors=31, metric="minkowski", weights="distance")
        elif classifier == "svm":
            clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        elif classifier == "rf":
            clf = RandomForestClassifier(max_features="log2", max_depth=10, min_samples_leaf=5, n_estimators=300)
        elif classifier == "cnn":
            clf = TimeSeriesCNNClassifier()
        elif classifier == "transformer":
            clf = TimeSeriesTransformerClassifier()
        elif classifier == "xgb":
            clf = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, min_child_weight=5, subsample=0.8, eval_metric="mlogloss", verbosity=0)

        train_fs = 500 if category == "eeg" else 100
        
        pipeline, feature_keys, metadata = train_reidentification_classifier(
            ref_attacker, fs=train_fs, classifier=clf, using_feature=use_feature, category=category,
            robust_features=robust_features, n_features=n_features,
        )

        print(f"Classifier: {clf.__class__.__name__}, trained on {len(ref_attacker)} subjects")

        # Build test feature matrix now that feature_keys is known
        X_test_rows = []
        for ts in ts_raw_test:
            if use_feature:
                if category == "ecg":
                    feats = extract_ecg_features(ts, fs=100)
                elif category == "eeg":
                    feats = extract_eeg_features(ts, fs=500, robust=robust_features)
                X_test_rows.append([feats[k] for k in feature_keys])
            else:
                X_test_rows.append(ts)

        if not X_test_rows:
            print("No test data found under", base_root)
            return None
        else:
            X_test = np.array(X_test_rows, dtype=np.float64)
            test_labels = np.array(labels, dtype=np.int64)

            # Replace NaN/Inf with column medians (mirrors reidentification_svm)
            for col in range(X_test.shape[1]):
                bad = ~np.isfinite(X_test[:, col])
                if bad.any():
                    X_test[bad, col] = np.nanmedian(X_test[~bad, col]) if not np.all(bad) else 0.0

            if plot and use_feature and feature_keys:
                tag = f"{dataset}_{'ipopt' if ipopt else 'fake'}"
                plot_feature_distributions(
                    X_test, feature_keys, labels=test_labels,
                    title=f"Test-set feature distributions — {tag}",
                    out=f"feature_distributions_{tag}.png",
                )
                plot_individual_separation(
                    X_test, feature_keys, test_labels,
                    title=f"Individual separation — {tag}",
                    out_prefix=f"separation_{tag}",
                )

            y_pred = pipeline.predict(X_test)

            # Also predict with the negated signal (symmetric solution)
            X_test_neg_rows = []
            for ts in ts_raw_test:
                if use_feature:
                    if category == "ecg":
                        feats_neg = extract_ecg_features(normalize(-ts), fs=100)
                    elif category == "eeg":
                        feats_neg = extract_eeg_features(normalize(-ts), fs=500, robust=robust_features)
                    X_test_neg_rows.append([feats_neg[k] for k in feature_keys])
                else:
                    X_test_neg_rows.append(-ts)
            X_test_neg = np.array(X_test_neg_rows, dtype=np.float64)
            for col in range(X_test_neg.shape[1]):
                bad = ~np.isfinite(X_test_neg[:, col])
                if bad.any():
                    X_test_neg[bad, col] = np.nanmedian(X_test_neg[~bad, col]) if not np.all(bad) else 0.0
            y_pred_neg = pipeline.predict(X_test_neg)

            hit_mask  = (y_pred == test_labels) | (y_pred_neg == test_labels)
            # hit_mask  = (y_pred == test_labels)
            correct   = hit_mask.sum()
            incorrect = len(y_pred) - correct
            accuracy  = correct / len(y_pred)
            print(f"True  : {correct}")
            print(f"False : {incorrect}")
            print(f"Accuracy : {accuracy:.4f}")

            if verbose:
                print(f"\n--- Per-test-patient ({classifier}) attribution ---")
                for i, ptb_id in enumerate(test_labels):
                    hit = "✓" if hit_mask[i] else "✗"
                    if hit_mask[i]:
                        print(f"#{i} (class {ptb_id:3d})  → predicted #{y_pred[i]} / {y_pred_neg[i]}  {hit}")
            accuracy = accuracy

    # Using distance method
    else:
        fs_dist = 500 if category == "eeg" else 100

        if use_feature:
            # Extract features from every reference segment, then fit a cross-patient
            # StandardScaler on one representative per person (mirrors conformal_prediction).
            dist_feature_keys = None
            ref_feat = []
            for ts_person_segs in ref_attacker:
                person_vecs = []
                for seg in ts_person_segs:
                    f = extract_eeg_features(seg, fs=fs_dist, robust=robust_features) if category == "eeg" else extract_ecg_features(seg, fs=fs_dist)
                    if dist_feature_keys is None:
                        dist_feature_keys = list(f.keys())
                    person_vecs.append(np.array([f[k] for k in dist_feature_keys], dtype=np.float64))
                ref_feat.append(person_vecs)

            n_refs_pp = len(ref_feat[0])
            all_ref = np.array([v for person in ref_feat for v in person], dtype=np.float64)
            for col in range(all_ref.shape[1]):
                bad = ~np.isfinite(all_ref[:, col])
                if bad.any():
                    all_ref[bad, col] = np.nanmedian(all_ref[~bad, col]) if not np.all(bad) else 0.0
            scaler_dist = StandardScaler().fit(all_ref[::n_refs_pp])
            all_ref_scaled = scaler_dist.transform(all_ref)
            ref_feat = [
                [all_ref_scaled[i * n_refs_pp + j] for j in range(n_refs_pp)]
                for i in range(len(ref_feat))
            ]

            def _feat_vec(ts):
                f = extract_eeg_features(ts, fs=fs_dist, robust=robust_features) if category == "eeg" else extract_ecg_features(ts, fs=fs_dist)
                v = np.array([f[k] for k in dist_feature_keys], dtype=np.float64)
                v[~np.isfinite(v)] = 0.0
                return scaler_dist.transform(v.reshape(1, -1)).flatten()

            ref_for_dist  = ref_feat
            test_vecs     = [_feat_vec(ts) for ts in ts_raw_test]
            # Mirror conformal_prediction: negate and min-max normalize the feature vector
            # (not the raw signal) so the two codepaths are identical.
            test_vecs_neg = [normalize(-v) for v in test_vecs]
        else:
            ref_for_dist  = ref_attacker
            test_vecs     = ts_raw_test
            test_vecs_neg = [normalize(-ts) for ts in ts_raw_test]

        # Print per feature distance
        if use_feature:
            for id_feat in range(len(dist_feature_keys)):
                feature_name = dist_feature_keys[id_feat]
                dist_feature = []
                for id_test_vec in range(len(test_vecs)):
                    ref_list = ref_for_dist[id_test_vec]
                    dist = np.min([abs(float(test_vecs[id_test_vec][id_feat]) - float(ref[id_feat])) for ref in ref_list])
                    dist_feature.append(dist)
                print(f"{feature_name}: {np.mean(dist_feature)}")

        test_labels = np.array(test_labels, dtype=np.int64)

        def _rank(sorted_cands, true_label):
            return next((r for r, (cid, _) in enumerate(sorted_cands) if cid == true_label), len(sorted_cands))

        ranks_d = []
        if metric == "pcc":
            metric_func = pearson_correlation
        elif metric == "euclidean":
            metric_func = euclidean
        elif metric == "dtw":
            metric_func = dtw_distance
        else:
            print("Unknown metric function name, using euclidean")
            metric_func = euclidean
        for ts_vec, ts_vec_neg, true_lbl in zip(test_vecs, test_vecs_neg, test_labels):
            print(true_lbl)
            row_pos, row_neg = [], []
            for candidate_refs in ref_for_dist:
                if metric_func.__name__ == "pearson_correlation":
                    v = [metric_func(ts_vec,     r) for r in candidate_refs]; row_pos.append(np.max(v) if v else float("nan"))
                    v = [metric_func(ts_vec_neg, r) for r in candidate_refs]; row_neg.append(np.max(v) if v else float("nan"))
                else:
                    v = [metric_func(ts_vec,     r) for r in candidate_refs]; row_pos.append(np.min(v) if v else float("nan"))
                    v = [metric_func(ts_vec_neg, r) for r in candidate_refs]; row_neg.append(np.min(v) if v else float("nan"))
            if metric_func.__name__ == "pearson_correlation":
                sc_pos = sorted(enumerate(row_pos), key=lambda x: x[1],reverse=True)
                sc_neg = sorted(enumerate(row_neg), key=lambda x: x[1], reverse=True)
            else:
                sc_pos = sorted(enumerate(row_pos), key=lambda x: x[1])
                sc_neg = sorted(enumerate(row_neg), key=lambda x: x[1])
            ranks_d.append(min(_rank(sc_pos, true_lbl), _rank(sc_neg, true_lbl)))
            print(min(_rank(sc_pos, true_lbl), _rank(sc_neg, true_lbl)))

        ranks_d   = np.array(ranks_d, dtype=np.int64)
        hit_mask_d = ranks_d == 0
        correct_d  = hit_mask_d.sum()
        accuracy_d = correct_d / len(ranks_d)
        top2_d     = (ranks_d <= 1).mean()
        print(f"\n--- Distance-based ({metric}) re-identification ---")
        print(f"True (rank=0)  : {correct_d}")
        print(f"False          : {len(ranks_d) - correct_d}")
        print(f"Accuracy @1    : {accuracy_d:.4f}  (= CDF at rank 0, comparable to conformal_prediction mean(ranks==0))")
        print(f"Accuracy @2    : {top2_d:.4f}  (= CDF at rank 1, what plot_paper threshold=0.05 annotates)")

        if verbose:
            print(f"\n--- Per-test-patient {metric} attribution ---")
            for i, ptb_id in enumerate(test_labels):
                hit = "✓" if hit_mask_d[i] else "✗"
                if hit_mask_d[i]:
                    print(f"#{i} (class {ptb_id:3d})  → rank {ranks_d[i]}  {hit}")
        accuracy = accuracy_d
    return accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-test-patient attribution breakdown")
    parser.add_argument("--mp", action="store_true", help="Use MP to reidentify")
    parser.add_argument("--category", type=str, default="ecg", help="Dataset category")
    parser.add_argument("--dataset", type=str, default="ptbxl", help="Evaluate on dataset")
    parser.add_argument("--ipopt", action="store_true", help="Use solution after ipopt")
    parser.add_argument("--classifier", type=str, default="svm", choices=["svm", "knn", "rf", "cnn", "transformer", "xgb"], help="The type of classifier used")
    parser.add_argument("--feature", action="store_true", help="Using feature extraction")
    parser.add_argument("--distance", action="store_true", help="Reidentify via nearest-neighbour distance instead of a classifier")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "dtw", "pcc", "eeg"], help="Distance metric for --distance mode (eeg = band-power + IAF feature space, recommended for single-channel EEG)")
    parser.add_argument("--original", action="store_true", help="Using original time series to reidentify")
    parser.add_argument("--resample", type=int, default=None, metavar="HZ",
                        help="Resample signals to this rate (Hz) before processing")
    parser.add_argument("--plot", action="store_true",
                        help="Save a KDE grid of per-feature distributions for the test set")
    parser.add_argument("--robust", action="store_true",
                        help="Use only IPOPT-robust EEG features (iaf, entropy, spectral_entropy, offset, exponent, ...)")
    parser.add_argument("--n-features", type=int, default=None, metavar="K",
                        help="Keep the K best features by Fisher score (default: use all)")
    parser.add_argument("--restEO", action="store_true",
                        help="Add eyes-open (restEO) segments to the reference enrollment (tdbrain only)")

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
        elif args.dataset == "tdbrain":
            base_root = "src/results/ipopt/eeg/"
    else:
        if args.dataset == "arrhythmia":
            base_root = "test/results/ecg_arrhythmia/"
        elif args.dataset == "ptbxl":
            base_root = "src/results/baseline/ptbxl/ptbxl/"
        elif args.dataset == "arrhythmia_xl":
            base_root = "test/results/ecg_arrhythmia_xl/"
        elif args.dataset == "ltdb":
            base_root = "test/results/ecg_ltdb_100/"
        elif args.dataset == "tdbrain":
            base_root = "src/results/baseline/eeg/tdbrain/"
            # base_root = "src/results/ipopt/eeg/"

    # if args.original and args.dataset == "tdbrain":
    #     base_root = os.path.join("data", "TDBRAIN-dataset")

    acc = reidentification_attack(base_root, 500, args.ipopt, args.dataset, args.category, args.mp, args.classifier, args.feature, args.original, args.distance, args.metric, args.verbose, resample_hz=args.resample, plot=args.plot, robust_features=args.robust, n_features=args.n_features)
    print(acc)

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
    #         # ts = np.concatenate([mp[:, ke_data0], mp[:, 1]])
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