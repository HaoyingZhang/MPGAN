import numpy as np
import os, sys, json
from scipy import stats
import matplotlib.pyplot as plt
import stumpy
import matplotlib
from src.utils_matrix_profile import build_mp_embedding
from dtaidistance import dtw
import wfdb
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from collections import Counter
import pandas as pd
from ecg_features import extract_ecg_features_bis
from sklearn.preprocessing import StandardScaler

def compute_loss_from_folder(base_folder, loss_function, m=200, epsilon=0.7, stat="mean"):
    mse_list = []
    for folder in os.listdir(base_folder):
        eval_folder = os.path.join(base_folder, folder)
        if os.path.isdir(eval_folder) and folder.startswith("ecg_"):
            res_file = os.path.join(eval_folder, "results.json")
            with open(res_file) as json_data:
                d = json.load(json_data)
                # ts_original = np.array(d["time_series"])
                ts_original = np.array(d["data"])
                # ts_fake = np.array(d["solutions"][0])
                ts_fake = np.array(d["fake_data"])
                if loss_function.__name__ == "partial_pearson_correlation":
                    loss = loss_function(ts_original, ts_fake, m=m)
                elif (loss_function.__name__ == "partial_pearson_correlation") or (loss_function.__name__ == "partial_rmse"):
                    loss = loss_function(ts_original, ts_fake, m=m, epsilon=epsilon)
                else:
                    loss = loss_function(ts_original, ts_fake, epsilon)
                mse_list.append(loss)
    if stat == "mean":
        return(np.mean(mse_list))
    if stat== "max":
        return(np.max(mse_list))
    if stat=="min":
        return(np.min(mse_list))
    
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

def partial_pearson_correlation(x, y, m, epsilon=None):
    n_patterns = len(x) - m + 1
    pccs = []
    for i in range(n_patterns):
        pcc = pearson_correlation(x[i:i+m], y[i:i+m])
        pccs.append(pcc)
    pcc = np.max(pccs)
    if epsilon is not None:
        return round(pcc, 1)>=epsilon
    else:
        return pcc
    
def mse(x, y):
    return ((x-y)**2).mean()

def rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.mean((x - y) ** 2))

def rmse_inv_check(list1, list2, epsilon=None):
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)

    n1 = normalize(list1)
    n2 = normalize(list2)
    n2_inv = normalize(-list2)

    rmse = np.min([np.sqrt(np.mean((n1 - n2) ** 2)), np.sqrt(np.mean((n1 - n2_inv) ** 2))])
    if epsilon is not None:
        return round(rmse, 1) <= epsilon
    else:
        return rmse

def partial_rmse(x, y, m, epsilon=None):
    n_patterns = len(x) - m + 1
    pccs = []
    for i in range(n_patterns):
        pcc = rmse_inv_check(x[i:i+m], y[i:i+m])
        pccs.append(pcc)
    rmse = np.min(pccs)
    if epsilon is not None:
        return round(rmse, 1)<=epsilon
    else:
        return rmse


def zdtw(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return dtw(a, b)

def ts_entropy(
    ts,
    m=100,
    distance_metric=None,
    epsilon=0.1,
    normalize_entropy=True
):
    """
    Subsequence-level pattern entropy via epsilon-clustering.

    ts: 1D time series
    m: subsequence length
    distance_metric: callable(x, y) -> distance
                     (default: z-normalized Euclidean)
    epsilon: distance threshold
    """

    ts = np.asarray(ts, dtype=np.float64)

    if len(ts) < 2 * m:
        raise ValueError(
            f"Not enough patterns of length {m} in time series"
        )

    # Default distance: z-normalized Euclidean
    if distance_metric is None:
        def distance_metric(a, b):
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            return np.linalg.norm(a - b)

    nb_patterns = len(ts) // m
    prototypes = []   # representative subsequences
    counts = []       # counts per prototype

    for i in range(nb_patterns):
        current = ts[i*m:(i+1)*m]

        if len(prototypes) == 0:
            prototypes.append(current)
            counts.append(1)
            continue

        # Compute distances to existing prototypes
        dists = np.array([
            distance_metric(current, p)
            for p in prototypes
        ])

        j = np.argmin(dists)

        if dists[j] < epsilon:
            counts[j] += 1
        else:
            prototypes.append(current)
            counts.append(1)
    # print(len(counts))

    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / counts.sum()

    H = entropy(probs)  # Shannon entropy (natural log)

    if normalize_entropy:
        H /= np.log(len(probs))

    return H


def parse_similar_mp_folders(base_folder, m=100):
    """
    Parse all ecg_<number> subfolders and return those where the matrix profile
    of the solution is similar to the matrix profile of the original time series.

    Similarity criteria:
      - Pearson correlation of MPD (matrix profile distances) > 0.6
      - Accuracy of MPI (matrix profile indices) > 100/400

    Keys tried:
      - Solution: "fake_data" first, then "solutions"[0]
      - Original: "data" first, then "time_series"

    :param base_folder: path to the folder containing ecg_* subfolders
    :param m: subsequence length for stumpy.stump
    :return: list of folder names that pass both similarity criteria
    """
    matching_folders = []

    for folder in os.listdir(base_folder):
        eval_folder = os.path.join(base_folder, folder)
        if not (os.path.isdir(eval_folder) and folder.startswith("ecg_")):
            continue

        res_file = os.path.join(eval_folder, "results.json")
        if not os.path.exists(res_file):
            continue

        with open(res_file) as f:
            d = json.load(f)

        # Resolve original time series
        if "data" in d:
            ts_original = np.array(d["data"], dtype=np.float64)
        elif "time_series" in d:
            ts_original = np.array(d["time_series"], dtype=np.float64)
        else:
            continue

        # Resolve solution time series
        if "fake_data" in d:
            ts_solution = np.array(d["fake_data"], dtype=np.float64)
        elif "solutions" in d and len(d["solutions"]) > 0:
            ts_solution = np.array(d["solutions"][0], dtype=np.float64)
        else:
            continue

        # Compute matrix profiles
        mp_orig = stumpy.stump(ts_original, m=m)
        mp_sol = stumpy.stump(ts_solution, m=m)

        mpd_orig = mp_orig[:, 0].astype(np.float64)
        mpi_orig = mp_orig[:, 1].astype(int)
        mpd_sol = mp_sol[:, 0].astype(np.float64)
        mpi_sol = mp_sol[:, 1].astype(int)

        # Criterion 1: Pearson correlation of MPD > 0.6
        if np.std(mpd_orig) < 1e-8 or np.std(mpd_sol) < 1e-8:
            mpd_corr = 0.0
        else:
            mpd_corr = pearson_correlation(mpd_orig, mpd_sol)

        if mpd_corr < 0.6:
            continue

        # Criterion 2: MPI accuracy > 100/400
        n = min(len(mpi_orig), len(mpi_sol))
        mpi_accuracy = np.sum(mpi_orig[:n] == mpi_sol[:n]) / n
        if mpi_accuracy <= 100 / 400:
            continue

        matching_folders.append(folder)

    return matching_folders


def plot_rank_distribution(
    ranks,
    save_path,
    xlabel="Rank range",
    ylabel="Count",
    title="Rank Distribution for person 1",
    ccdf=False,
    threshold=0.75
):
    ranks = np.asarray(ranks, dtype=int)

    if np.any(ranks < 0):
        raise ValueError("Ranks must be >= 0")

    if ccdf:
        sorted_ranks = np.sort(ranks)
        n = len(sorted_ranks)
        ccdf_values = np.arange(1, n + 1) / n

        fig, ax = plt.subplots()
        ax.plot(sorted_ranks, ccdf_values)

        # Find first point where cumulative accuracy >= threshold
        idx = np.searchsorted(ccdf_values, threshold, side="left")
        idx = min(idx, n - 1)
        x_cross = sorted_ranks[idx]
        y_cross = ccdf_values[idx]

        ax.axhline(y=y_cross, color="red", linestyle="--", linewidth=1)
        ax.axvline(x=x_cross, color="red", linestyle="--", linewidth=1)
        ax.annotate(
            f"({x_cross}, {y_cross:.2f})",
            xy=(x_cross, y_cross),
            xytext=(x_cross + max(sorted_ranks) * 0.03, y_cross - 0.06),
            fontsize=9,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

        ax.set_xlabel("Rank")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title(title)
        fig.savefig(save_path)
        print(f"Plot saved in {save_path}")
        return

    rank_counts = Counter(ranks)
    max_rank = ranks.max()

    # Define fixed rank bins (5 bars)
    bins = [0, 1, 10, 100, 1000, max_rank + 1]

    # Count occurrences per bin
    bin_counts = []
    bin_labels = []

    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i + 1]
        c = sum(v for k, v in rank_counts.items() if start <= k < end)
        bin_counts.append(c)

        if end == max_rank + 1:
            bin_labels.append(f"{start}+")
        else:
            bin_labels.append(f"{start}–{end-1}")

    # Plot
    fig = plt.figure()
    bars = plt.bar(bin_labels, bin_counts)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Annotate counts
    for bar, count in zip(bars, bin_counts):
        if count > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=9
            )

    # plt.show()
    fig.savefig(save_path)
    print(f"Plot saved in {save_path}")

def normalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())

def znormalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.mean()) / (time_series.std() + 1e-8)

def conformal_prediction(n, m, base_folder_test, dataset, ref_index=[0], using_features=False, metric=euclidean):
    """
    Conformal prediction-based re-identification attack.

    For each reconstructed time series in base_folder_test, ranks all candidate
    patients by similarity and returns the rank of the true original patient.

    :param n: Length of time series
    :param m: Subsequence length
    :param base_folder_test: Base folder containing per-result subfolders
    :param dataset: Dataset used ("ptbxl" or "arrhythmia")
    :param ref_index: List of start indices for reference subsequences
    :param using_features: (unused) Enable using features instead of raw time series
    :param metric: Distance callable(x, y) -> float; lower = more similar (default: euclidean)
    :return: List of 0-indexed ranks of the true patient among all candidates
    """
    if n - m + 1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    feature_keys = []
    # Load reference subsequences for each patient in the dataset
    ref_attacker = []

    if dataset == "ptbxl":
        list_patient = pd.read_csv("data/physionet.org/files/ptbxl_database.csv")["filename_lr"]
        files = list_patient[21000:21200]
    elif dataset in("arrhythmia", "arrhythmia_xl"):
        with open("data/physionet.org/files/ecg-arrhythmia/records100/RECORDS", "r") as f:
            list_patient = f.read().splitlines()
        files = [os.path.join("data/physionet.org/files/ecg-arrhythmia/records100/", file + ".npy") for file in list_patient]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected 'ptbxl' or 'arrhythmia'.")

    # print(f"Will take {n} points from indices: {ref_index}")

    for file in files:
        if dataset == "ptbxl":
            record = wfdb.rdrecord(os.path.join("data/physionet.org/files/", file))
            signal = record.p_signal[:, 0].astype(np.float64)
        elif dataset in ("arrhythmia","arrhythmia_xl"):
            signal = np.load(file)

        ref_attacker_person = [normalize(signal[index:index + n]) for index in ref_index]
        if using_features:
            ts_ref_features = [extract_ecg_features_bis(ts_ref, fs=100) for ts_ref in ref_attacker_person]
            if len(feature_keys) == 0:
                feature_keys = list(ts_ref_features[0].keys())
            ref_attacker_person = [np.array([ts_ref_feat[key] for key in feature_keys]) for ts_ref_feat in ts_ref_features]

        ref_attacker.append(ref_attacker_person)

    # Fit a cross-patient StandardScaler on reference features so that each
    # feature dimension is normalized by its variance across patients
    # (mirrors the StandardScaler used in the SVM/1-NN pipeline).
    if using_features:
        n_persons = len(ref_attacker)
        n_refs = len(ref_attacker[0])
        # Stack all vectors into one matrix: shape (n_persons * n_refs, n_features)
        all_matrix = np.array(
            [vec for person in ref_attacker for vec in person], dtype=np.float64
        )
        # Replace NaN/Inf with column medians before fitting or transforming
        for col in range(all_matrix.shape[1]):
            bad = ~np.isfinite(all_matrix[:, col])
            if bad.any():
                all_matrix[bad, col] = np.nanmedian(all_matrix[~bad, col]) if not np.all(bad) else 0.0
        # Fit on one representative vector per patient (first ref_index)
        ref_matrix = all_matrix[::n_refs]
        scaler = StandardScaler()
        scaler.fit(ref_matrix)
        # Transform all vectors at once, then rebuild ref_attacker
        all_matrix_scaled = scaler.transform(all_matrix)
        ref_attacker = [
            [all_matrix_scaled[i * n_refs + j] for j in range(n_refs)]
            for i in range(n_persons)
        ]
    plot_test_points_distribution(ref_attacker, feature_keys)

    # Load reconstructed time series from result subfolders
    ts_inverse = []

    result_folders = sorted(
        [d for d in os.listdir(base_folder_test) if os.path.isdir(os.path.join(base_folder_test, d)) and d.startswith("ecg_")],
        key=lambda d: int(d.split("ecg_")[1])
    )
    print(result_folders)

    for folder in result_folders:
        result_path = os.path.join(base_folder_test, folder, "results.json")
        with open(result_path, "r") as f:
            res = json.load(f)
        ts_inv = normalize(np.array(res["solutions"][0]))
        # ts_inv = normalize(np.array(res["fake_data"]))
        if using_features:
            ts_inv_features = extract_ecg_features_bis(ts_inv, fs=100)
            if len(feature_keys) == 0:
                feature_keys = list(ts_inv_features.keys())
            ts_inv = np.array([ts_inv_features[feat] for feat in feature_keys])
            # Apply the same cross-patient scaler fitted on reference features
            ts_inv = scaler.transform(ts_inv.reshape(1, -1)).flatten()

        ts_inverse.append(ts_inv)
        
    # For each reconstructed ts, rank candidates by distance to reference subsequences
    rank = []
    for id_ts, ts_test in enumerate(ts_inverse):
        # Mean distance across all reference indices for each candidate patient
        distances = []
        for candidate_refs in ref_attacker:
            dist = np.mean([
                metric(ts_test, ref_subseq)
                for ref_subseq in candidate_refs
            ])
            distances.append(dist)

        # Sort ascending: lowest distance = most similar = rank 0
        if metric.__name__ == "pearson_correlation":
            sorted_candidates = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)
        else:
            sorted_candidates = sorted(enumerate(distances), key=lambda x: x[1])

        # Find the rank of the true original patient (id_ts)
        id_label = int(id_ts / 5) if dataset=="arrhythmia_xl" else id_ts
        rank_id = next(
            r for r, (cand_id, _) in enumerate(sorted_candidates)
            if cand_id == id_label
        )
        rank.append(rank_id)
        if rank_id == 0:
            print(id_ts)
        # print(f"{id_ts}:{rank_id}")

    return rank

def plot_test_points_distribution(ref_attacker, feature_keys=None, save_path="test_points_distribution.png"):
    matrix = np.array([person[0] for person in ref_attacker])  # (n_patients, n_dims)
    n_patients, n_dims = matrix.shape

    colors = plt.cm.tab20(np.linspace(0, 1, n_patients))

    fig, ax = plt.subplots(figsize=(max(10, n_dims // 2), 5))
    ax.boxplot(matrix, positions=range(n_dims), showfliers=False)

    for i in range(n_patients):
        ax.scatter(range(n_dims), matrix[i], alpha=0.6, s=15, color=colors[i], label=f"class {i}")

    if feature_keys:
        ax.set_xticks(range(n_dims))
        ax.set_xticklabels(feature_keys, rotation=90, fontsize=7)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.set_title(f"Distribution of test points ({n_patients} patients, 1 point/class)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=6, ncol=max(1, n_patients // 20), markerscale=2)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")



if __name__ == "__main__":
    # for id_person in range(1,2):
    #     rank_0 = conformal_prediction(500, 100, f"src/results/baseline/n500m100/person{id_person}", 20000, [id_person])
    #     np.save(f"outputs/n200m10/disable_validation/rank_{id_person}.npy", rank_0)
    #     plot_rank_distribution(rank_0, f"outputs/n200m10/disable_validation/conformal_{id_person}.png", title=f"Rank Distribution for person {id_person}")
    # base_test_folder = "src/results/baseline/2026-02-12_23:48:26"
    # base_test_folder = "/home/haoying/Documents/MPGAN/src/results/baseline/2026-03-04_10:10:37"
    # base_test_folder = "src/results/baseline/2026-03-19_01:02:19/"
    # base_test_folder = "test/results/2026-03-30_10:54:17"
    # base_test_folder = "/home/haoying/Documents/MPGAN/src/results/ipopt/arrhythmia/"
    # base_test_folder = "/home/haoying/Documents/MPGAN/src/results/ipopt/ptbxl/"
    # base_test_folder = "/home/haoying/Documents/MPGAN/test/results/ecg_arrhythmia/"
    # base_test_folder = "/home/haoying/Documents/MPGAN/test/results/ecg_arrhythmia_xl/"
    # base_test_folder = "/home/haoying/Documents/MPGAN/test/results/ecg_ltdb_128/"
    # base_test_folder = "/home/haoying/Documents/MPGAN/src/results/ipopt/ltdb_128"
    # base_test_folder = "/home/haoying/Documents/MPGAN/src/results/baseline/2026-04-17_14:04:40"
    base_test_folder = "/home/haoying/Documents/MPGAN/src/results/baseline/ptbxl"
    # base_test_folder = "src/results/ipopt/arrhythmia_xl"

    loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="mean")
    print(f"PCC mean : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=0.7, stat="mean")
    print(f"PCC 0.7 : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="max")
    print(f"PCC max : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, partial_pearson_correlation, m=20, epsilon=0.7, stat="mean")
    print(f"Partial PCC 0.7 : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=None, stat="mean")
    print(f"RMSE : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=0.1, stat="mean")
    print(f"RMSE 0.1 : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=None, stat="min")
    print(f"RMSE min : {loss_test}")
    loss_test = compute_loss_from_folder(base_test_folder, partial_rmse, m=20, epsilon=0.1, stat="mean")
    print(f"Partial RMSE 0.1 : {loss_test}")

    # similar_folders = parse_similar_mp_folders(base_test_folder, m=100)
    # print(similar_folders)
    # ranks = conformal_prediction(n=500, m=100, base_folder_test=base_test_folder, ref_index=[0], using_features=True, dataset="arrhythmia_xl", metric=euclidean)
    # plot_rank_distribution(ranks, "rank_distribution_arrhythmia_xl.png", title="Rank Distribution", ccdf=True)



    # for id_p in range(1):
        # print(f"person{id_p}")
        # base_test_folder = f"src/results/baseline/n200m10/disable_validation/person{id_p}"
        # loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="mean")
        # print(f"PCC mean : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=0.7, stat="mean")
        # print(f"PCC 0.7 : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="max")
        # print(f"PCC max : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, partial_pearson_correlation, m=20, epsilon=0.7, stat="mean")
        # print(f"Partial PCC 0.7 : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=None, stat="mean")
        # print(f"RMSE : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=0.1, stat="mean")
        # print(f"RMSE 0.1 : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=None, stat="min")
        # print(f"RMSE min : {loss_test}")
        # loss_test = compute_loss_from_folder(base_test_folder, partial_rmse, m=20, epsilon=0.1, stat="mean")
        # print(f"Partial RMSE 0.1 : {loss_test}")
