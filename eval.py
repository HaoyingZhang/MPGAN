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
from scipy.stats import entropy
from collections import Counter

def compute_loss_from_folder(base_folder, loss_function, m=200, epsilon=0.7, stat="mean"):
    mse_list = []
    for folder in os.listdir(base_folder) :
        eval_folder = os.path.join(base_folder, folder)
        if os.path.isdir(eval_folder):
            res_file = os.path.join(eval_folder, "results.json")
            with open(res_file) as json_data:
                d = json.load(json_data)
                ts_original = np.array(d["data"])
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

def rmse_inv_check(list1,list2, epsilon=None):
    mean_2 = np.mean(list2)+np.std(list2)
    list2_inv = np.array([2*mean_2-x for x in list2])
    rmse = np.min([np.sqrt(np.mean((list1 - list2) ** 2)), np.sqrt(np.mean((list1 - list2_inv)**2))])
    if epsilon is not None:
        return round(rmse, 1)<=epsilon
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


def plot_rank_distribution(
    ranks,
    save_path,
    xlabel="Rank range",
    ylabel="Count",
    title="Rank Distribution for person 1",
):
    ranks = np.asarray(ranks, dtype=int)

    if np.any(ranks < 0):
        raise ValueError("Ranks must be >= 0")

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

def conformal_prediction(n,m, base_folder, n_ts, train_id):
    max_index_list = [10828800, 6420480, 10997760, 9454080, 9753600, 10252800, 10237440]
    max_index = np.min([max_index_list[ind] for ind in train_id])
    print(max_index)
    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    list_patient = [14046, 14134, 14149, 14157, 14172, 14184, 15814]
    data_dir = "data/physionet.org/files/ltdb/1.0.0/"
    files = sorted([os.path.join(data_dir, str(list_patient[i])) for i in train_id])
    n_person_training = len(train_id)
    n_ts_per_person_train = n_ts // n_person_training
    n_ts_per_person_test = 200
    np.random.seed(2025)
    rng = np.random.default_rng(2025)
    max_start = max_index - n + 1
    assert max_start > 0, "Signal shorter than window length"

    n_samples = n_ts_per_person_test + n_ts_per_person_train
    max_possible = max_index // n

    if n_samples > max_possible:
        print("Not enough room for spaced sampling, overlapped time series will be used")
        indices_ts = np.random.randint(0, max_start, size=n_samples)
    else:
        print("No overlapped time series used")
        candidates = np.arange(0, max_start, n, dtype=np.int64)
        indices_ts = rng.choice(candidates, size=n_samples, replace=False)

    indices_ts_train = indices_ts[:n_ts_per_person_train]

    ts_train = []
    ts_test = []
    ts_inverse = []

    for file in files:
        record = wfdb.rdrecord(file)
        signal = record.p_signal[:, 0].astype(np.float32, copy=False)

        for start_idx in indices_ts_train:
            ts = signal[start_idx : start_idx + n]
            ts_norm = normalize(ts).astype(np.float32, copy=False)
            ts_train.append(ts_norm)

    for folder in os.listdir(base_folder) :
        eval_folder = os.path.join(base_folder, folder)
        if os.path.isdir(eval_folder):
            res_file = os.path.join(eval_folder, "results.json")
            with open(res_file) as json_data:
                d = json.load(json_data)
                ts_original = np.array(d["data"])
                ts_fake = np.array(d["fake_data"])
                ts_test.append(ts_original)
                ts_inverse.append(ts_fake)

    ts_candidate = np.concatenate([ts_test, ts_train])
    rank = []
    for id_ts in range(len(ts_test)):
        ts_original = ts_test[id_ts]
        ts_inversed = ts_inverse[id_ts]
        pccs = [pearson_correlation(ts_inversed, ts) for ts in ts_candidate]
        pccs_sort = sorted(
            enumerate(pccs),
            key=lambda x: x[1],
            reverse=True
        )

        # Find the rank of the true original time series
        # (its index in ts_candidate is id_ts)
        rank_id = [idx for idx, (cand_id, _) in enumerate(pccs_sort)
                if cand_id == id_ts][0]
        print(rank_id)
        rank.append(rank_id)

    return(rank)

if __name__ == "__main__":
    for id_person in range(4,7):
        rank_0 = conformal_prediction(200, 10, f"src/results/baseline/n200m10/disable_validation/person{id_person}", 20000, [id_person])
        np.save(f"outputs/n200m10/disable_validation/rank_{id_person}.npy", rank_0)
        plot_rank_distribution(rank_0, f"outputs/n200m10/disable_validation/conformal_{id_person}.png", title=f"Rank Distribution for person {id_person}")

    # for id_p in range(7):
    #     print(f"person{id_p}")
    #     base_test_folder = f"src/results/baseline/n200m10/disable_validation/person{id_p}"
    #     loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="mean")
    #     print(f"PCC mean : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=0.7, stat="mean")
    #     print(f"PCC 0.7 : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="max")
    #     print(f"PCC max : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, partial_pearson_correlation, m=20, epsilon=0.7, stat="mean")
    #     print(f"Partial PCC 0.7 : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=None, stat="mean")
    #     print(f"RMSE : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=0.1, stat="mean")
    #     print(f"RMSE 0.1 : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, rmse_inv_check, m=20, epsilon=None, stat="min")
    #     print(f"RMSE min : {loss_test}")
    #     loss_test = compute_loss_from_folder(base_test_folder, partial_rmse, m=20, epsilon=0.1, stat="mean")
    #     print(f"Partial RMSE 0.1 : {loss_test}")
