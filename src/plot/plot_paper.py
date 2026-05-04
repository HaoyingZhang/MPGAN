import stumpy, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import json, sys, os
from collections import Counter
from scipy.spatial.distance import euclidean

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # Add root directory to path
from src.utils import rmse, pearson_correlation
from eval import conformal_prediction
from eval import compute_loss_from_folder, compute_utility_loss_from_folder
from reidentify import reidentification_attack
from features_extraction import extract_ecg_features, extract_eeg_features



def plot_three_ts_with_mp(ts_deepmp: np.ndarray, ts_ipopt: np.ndarray, ts_original: np.ndarray, ts_intermediate: np.ndarray|None, m: int, znorm: bool = True):
    """
    Plot three time series and their matrix profiles.

    Layout:
      - Top    : the three time series superposed (blue, red, gray dotted)
      - Bottom-left  : matrix profile distances for the three series
      - Bottom-right : matrix profile indices for the three series

    The gray series uses dotted lines throughout (time series and both MP plots).

    Parameters
    ----------
    ts_deepmp : np.ndarray
        First time series, plotted in blue.
    ts_ipopt : np.ndarray
        Second time series, plotted in red.
    ts_original : np.ndarray
        Third time series, plotted with dotted gray lines.
    m : int
        Subsequence length used to compute the matrix profile.
    znorm : bool
        Whether to use z-normalisation when computing the matrix profile.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # --- compute matrix profiles ---
    def _clean_mp(ts):
        mp = stumpy.stump(ts, m=m, normalize=znorm)
        result = mp[:, [0, 1]].astype(np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    mp_blue = _clean_mp(ts_deepmp)
    if ts_intermediate is not None:
        mp_red = _clean_mp(ts_intermediate)
    else:
        mp_red  = _clean_mp(ts_ipopt)
    mp_gray = _clean_mp(ts_original)

    rmse_mpd_red = rmse(mp_red[:, 0], mp_gray[:, 0])
    rmse_mpd_blue = rmse(mp_blue[:, 0], mp_gray[:, 0])

    distance_mpi_red = np.mean(mp_red[:, 1] == mp_gray[:, 1])
    distance_mpi_blue = np.mean(mp_blue[:, 1] == mp_gray[:, 1])

    pcc_ts_red = pearson_correlation(np.array(ts_ipopt), np.array(ts_original))
    pcc_ts_blue = pearson_correlation(np.array(ts_deepmp), np.array(ts_original))

    # --- layout ---
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_ts  = fig.add_subplot(gs[0, :])   # full-width top row
    ax_mpd = fig.add_subplot(gs[1, 0])   # bottom-left  : MP distance
    ax_mpi = fig.add_subplot(gs[1, 1])   # bottom-right : MP index

    # --- top: time series ---
    ax_ts.plot(ts_deepmp, color="blue",  label=f"DeepMP Solution (PCC = {pcc_ts_blue:.2f})", linewidth=1.2)
    if ts_intermediate is not None:
        ax_ts.plot(ts_intermediate, color="red", linestyle="dotted", label="IPOPT Intermediate Solution", linewidth=1.0)
    ax_ts.plot(ts_ipopt,  color="red",   label=f"IPOPT Solution (PCC = {pcc_ts_red:.2f})", linewidth=1.2)
    ax_ts.plot(ts_original, color="gray",  label="Ground Truth", linewidth=1.2)
    ax_ts.set_title("Time Series", fontsize=12)
    ax_ts.legend(fontsize=9)

    # --- bottom-left: MP distance ---
    ax_mpd.plot(mp_blue[:, 0], color="blue", label=f"MPD DeepMP (RMSE = {rmse_mpd_blue:.2f})", linewidth=1.2)
    ax_mpd.plot(mp_red[:, 0],  color="red",  label=f"MPD IPOPT (RMSE = {rmse_mpd_red:.2f})", linewidth=1.2)
    ax_mpd.plot(mp_gray[:, 0], color="gray", label="MPD Ground Truth", linewidth=1.2)
    ax_mpd.set_title("Matrix Profile Distance", fontsize=12)
    ax_mpd.legend(fontsize=9)

    # --- bottom-right: MP index ---
    x_blue = np.arange(len(mp_blue))
    x_red  = np.arange(len(mp_red))
    x_gray = np.arange(len(mp_gray))

    ax_mpi.scatter(x_blue, mp_blue[:, 1], color="blue", label=f"MPI DeepMP (Accuracy = {distance_mpi_blue:.2f})", s=4)
    ax_mpi.scatter(x_red,  mp_red[:, 1],  color="red",  label=f"MPI IPOPT (Accuracy = {distance_mpi_red:.2f})", s=4)
    ax_mpi.scatter(x_gray, mp_gray[:, 1], color="gray", label="MPI Ground Truth", s=4
                   , marker="."
                   )
    ax_mpi.set_title("Matrix Profile Index", fontsize=12)
    ax_mpi.legend(fontsize=9)

    return fig

def plot_rank_distribution(
    rank_map,
    save_path,
    title="Rank Distribution for person 1",
    threshold=0.75
):
    fig, ax = plt.subplots()

    for key in rank_map.keys():
        ranks = rank_map[key]
        ranks = np.asarray(ranks, dtype=int)

        if np.any(ranks < 0):
            raise ValueError("Ranks must be >= 0")

        sorted_ranks = np.sort(ranks)
        n = len(sorted_ranks)
        if n == 0:
            continue
        ccdf_values = np.arange(1, n + 1) / n
        ax.plot(sorted_ranks, ccdf_values, label=key)
        # Find first point where cumulative accuracy >= threshold
        idx = np.searchsorted(ccdf_values, threshold, side="left")
        idx = min(idx, n - 1)
        x_cross = sorted_ranks[idx]
        y_cross = ccdf_values[idx]
        ax.axhline(y=y_cross, color="red", linestyle="--", linewidth=1)
        ax.axvline(x=x_cross, color="red", linestyle="--", linewidth=1)
        if key == "MP":
            ax.annotate(
                f"({x_cross}, {y_cross:.2f})",
                xy=(x_cross, y_cross),
                xytext=(x_cross + max(sorted_ranks) * 0.03, y_cross - 0.06),
                fontsize=13,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )
        else:
            ax.annotate(
                f"({x_cross}, {y_cross:.2f})",
                xy=(x_cross, y_cross),
                xytext=(x_cross - max(sorted_ranks) * 0.03, y_cross + 0.1),
                fontsize=13,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

    ax.set_xlabel("Rank", fontsize=14)
    ax.set_ylabel("Cumulative RIR", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend()
    fig.savefig(save_path)
    print(f"Plot saved in {save_path}")
    return


def plot_feature_relative_error_boxplot(
    dataset_paths,
    category="ecg",
    fs=100,
    out="paper_figures/feature_relative_error_boxplot.png",
):
    """
    For each dataset in dataset_paths, load every {category}_{N}/results.json,
    compare features extracted from the reference signal ("time_series" key) and
    the IPOPT reconstruction ("solutions"[0]), compute the relative error per
    feature, and display side-by-side box plots — one colour per dataset.

    Parameters
    ----------
    dataset_paths : dict[str, str]
        {"dataset_name": "path/to/ipopt/results/folder/", ...}
    category : str
        Sub-folder prefix, e.g. "ecg".
    fs : int
        Sampling frequency passed to extract_ecg_features.
    out : str
        Output PNG path.
    """
    from matplotlib.lines import Line2D
    import matplotlib.ticker as ticker

    NAMED_COLORS = {"arrhythmia": "#E07B54", "ptbxl": "#5B84B1",
                    "ltdb": "#4CAF50", "tdbrain": "#9C59B6", "trajectory": "#F1C40F"}
    MARKER_POOL  = ["o", "D", "s", "^", "v", "P", "X", "*"]
    DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    n_datasets = len(dataset_paths)
    ds_list    = list(dataset_paths.keys())
    colors  = [NAMED_COLORS.get(ds, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
               for i, ds in enumerate(ds_list)]
    markers = [MARKER_POOL[i % len(MARKER_POOL)] for i in range(n_datasets)]

    # ------------------------------------------------------------------ #
    # 1. Collect per-dataset, per-feature relative errors
    # ------------------------------------------------------------------ #
    all_errors   = {}   # dataset -> {feat_name: [rel_err, ...]}
    feature_keys = None

    for dataset, base_root in dataset_paths.items():
        dirs = sorted(
            [d for d in os.listdir(base_root)
             if d.startswith(f"{category}_") and d.split("_")[1].isdigit()],
            key=lambda x: int(x.split("_")[1]),
        )
        errors = {}
        for d in dirs:
            json_path = os.path.join(base_root, d, "results.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)

            ref_raw = np.array(data.get("time_series", data.get("data")), dtype=np.float64)
            # rec_raw = np.array(data["solutions"][0], dtype=np.float64)
            rec_raw = np.array(data["smoothed"], dtype=np.float64)

            try:
                f_ref = extract_eeg_features(ref_raw, fs=fs)
                f_rec = extract_eeg_features(rec_raw, fs=fs)
            except Exception:
                continue

            if feature_keys is None:
                feature_keys = list(f_ref.keys())

            for k in feature_keys:
                rel_err = abs(f_ref[k] - f_rec[k]) / (abs(f_ref[k]) + 1e-8)
                errors.setdefault(k, []).append(float(rel_err))

        all_errors[dataset] = errors
        print(f"[{dataset}] loaded {len(dirs)} files, {sum(len(v) for v in errors.values())} feature values")

    if feature_keys is None:
        print("No data found — check dataset_paths.")
        return

    # ------------------------------------------------------------------ #
    # 2. Build the figure — narrow side-by-side boxes, single-column style
    # ------------------------------------------------------------------ #
    matplotlib.rcParams.update({"font.size": 14})

    box_w   = 0.22
    group_w = n_datasets * box_w + 0.18

    n_feats = len(feature_keys)

    fig, ax = plt.subplots(figsize=(max(10, n_feats * (0.35 + 0.12 * n_datasets)), 4.5))

    for di, (dataset, errors) in enumerate(all_errors.items()):
        color         = colors[di]
        marker        = markers[di]
        positions     = [i * group_w + di * box_w for i in range(n_feats)]
        data_per_feat = [errors.get(k, [np.nan]) for k in feature_keys]

        ax.boxplot(
            data_per_feat,
            positions=positions,
            widths=box_w * 0.9,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.8, linewidth=1.1),
            medianprops=dict(color="white", linewidth=2.2),
            flierprops=dict(marker=marker, markersize=3.5, alpha=0.55,
                            markerfacecolor=color, markeredgecolor=color,
                            markeredgewidth=0.4),
            whiskerprops=dict(linewidth=1.3, color=color),
            capprops=dict(linewidth=1.6, color=color),
            showfliers=True,
        )

        for x_pos, vals in zip(positions, data_per_feat):
            finite = [v for v in vals if np.isfinite(v)]
            if not finite:
                continue
            ax.text(x_pos, 10, f"{np.median(finite):.2g}",
                    ha="center", va="bottom", rotation=90,
                    fontsize=7, color="black", fontweight="bold")

    # x-ticks at group centre
    tick_pos = [i * group_w + (n_datasets - 1) * box_w / 2 for i in range(n_feats)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(feature_keys, rotation=45, ha="right", fontsize=13)
    ax.set_ylabel("Relative error", fontsize=15)
    ax.set_yscale("log")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.tick_params(axis="y", labelsize=13)

    legend_handles = [
        Line2D([0], [0], marker=markers[i], color="none",
               markerfacecolor=colors[i], markersize=8,
               label=ds, markeredgewidth=0)
        for i, ds in enumerate(ds_list)
    ]
    ncol = min(n_datasets, 4)
    ax.legend(handles=legend_handles, fontsize=12, loc="upper left",
              framealpha=0.85, edgecolor="#cccccc", ncol=ncol)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    # deepmp_sol_path = "src/results/baseline/ptbxl/ptbxl/ecg_188/results.json"
    # ipopt_sol_path = "src/results/ipopt/ptbxl/ecg_188/results.json"
    # deepmp_sol_path = "src/results/baseline/eeg/tdbrain/eeg_118/results.json"
    # ipopt_sol_path = "src/results/ipopt/eeg/eeg_118/results.json"

    # with open(deepmp_sol_path, "r") as f:
    #     res = json.load(f)
    #     ts_deepmp = res["fake_data"]
    #     ts_original = res["data"]
        
    # with open(ipopt_sol_path, "r") as f:
    #     res = json.load(f)
    #     ts_ipopt = res["solutions"][0]
    #     ts_smooth = res["smoothed"]
    
    # fig = plot_three_ts_with_mp(ts_deepmp, ts_smooth, ts_original, ts_ipopt, m=100, znorm=True)
    # fig.savefig("paper_figures/compare_ipopt_eeg_new.jpg")

    # Conformal prediction
    # base_test_folder = "src/results/ipopt/ptbxl/"
    # ranks = conformal_prediction(n=500, m=100, base_folder_test=base_test_folder, ref_index=[0], using_features=True, dataset="ptbxl", metric=euclidean)
    # ranks_mp = conformal_prediction(n=500, m=100, base_folder_test=base_test_folder, ref_index=[0], using_features=False, dataset="ptbxl", metric=euclidean, using_mp=True)
    # plot_rank_distribution({"TS reconstructed": ranks, "MP": ranks_mp}, "paper_figures/rank_distribution_ptbxl.png", title="Rank Distribution", threshold=0.66)


    # Robustness
    # eval_type = "mpi"
    # datasets = ["arrhythmia", "ptbxl"]
    
    # baseline_accuracy = {"arrhythmia":0.129, "ptbxl":0.045}
    # test_id = {"arrhythmia":[0,48], "ptbxl":[21200,21400]}
    # epsilon = [100.0,1000.0,10000.0,100000.0,500000.0,800000.0, 1000000.0,100000000.0]
    
    # root_path = os.path.join("src/results/baseline/ptbxl/")

    # results_path = os.path.join("src/results/baseline/ptbxl", f"eval_robustness_{eval_type}_results.json")
    # os.makedirs(os.path.dirname(results_path), exist_ok=True)
    # if os.path.exists(results_path):
    #     with open(results_path) as f:
    #         all_results = json.load(f)
    # else:
    #     all_results = {}

    # pcc_list = []
    # accuracy_list = []
    # utility_loss_list = []
    # for dataset in datasets:
    #     if dataset not in all_results:
    #         all_results[dataset] = {}
    #     dataset_res = []
    #     reidentify = []
    #     utility_loss = []
    #     for eps in epsilon:
    #         eps_key = str(eps)
    #         if eps_key not in all_results[dataset]:
    #             all_results[dataset][eps_key] = {"pcc": [], "acc": [], "utility_loss": []}
    #         elif "utility_loss" not in all_results[dataset][eps_key]:
    #             all_results[dataset][eps_key]["utility_loss"] = []
    #         existing_runs = len(all_results[dataset][eps_key]["acc"])
    #         for run in range(existing_runs, 5):
    #             cmd = ["python3", "test/eval_robustness.py", "-dataset", dataset, "-eps", str(eps),
    #                    "-n_ts", "200", "-n", "500", "-m", "100", "-c", "ecg", "-train_id", "0", "48",
    #                    "-test_id", str(test_id[dataset][0]), str(test_id[dataset][1]), "-g_model", "WillBeNamed",
    #                    "-mp_embedding", "-znorm", "-fill", "100", "-test", "-rob", eval_type ]
    #             subprocess.run(cmd, check=True)
    #             base_test_folder = os.path.join(root_path, dataset, eval_type, str(eps))
    #             loss_test = compute_loss_from_folder(base_test_folder, pearson_correlation, m=20, epsilon=None, stat="mean")
    #             accuracy = reidentification_attack(base_test_folder,
    #                                             ipopt=False,
    #                                             dataset=dataset,
    #                                             category="ecg",
    #                                             use_mp=False,
    #                                             classifier="svm",
    #                                             use_feature=True,
    #                                             use_original=False,
    #                                             distance=False)
    #             loss_utility = compute_utility_loss_from_folder(base_test_folder, stat="mean")
    #             all_results[dataset][eps_key]["pcc"].append(float(loss_test))
    #             all_results[dataset][eps_key]["acc"].append(float(accuracy))
    #             all_results[dataset][eps_key]["utility_loss"].append(float(loss_utility))
    #             with open(results_path, "w") as f:
    #                 json.dump(all_results, f, indent=2)
    #             print(f"[{dataset}] eps={eps} run {run+1}/5 — pcc={loss_test:.4f} acc={accuracy:.4f} utility loss={loss_utility:.4f}")
    #         dataset_res.append(np.mean(all_results[dataset][eps_key]["pcc"]))
    #         reidentify.append(np.mean(all_results[dataset][eps_key]["acc"]))
    #         utility_loss.append(np.mean(all_results[dataset][eps_key]["utility_loss"]))
    #     pcc_list.append(dataset_res)
    #     accuracy_list.append(reidentify)
    #     utility_loss_list.append(utility_loss)

    # def fmt_eps(v):
    #     exp = int(np.log10(v))
    #     coeff = round(v / 10**exp)
    #     return f"${coeff}\\times10^{{{exp}}}$"

    # eps_labels = [fmt_eps(e) for e in epsilon]

    # from matplotlib.ticker import PercentFormatter
    # matplotlib.rcParams.update({"font.size": 13})
    # x = np.arange(len(epsilon))
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # os.makedirs("paper_figures", exist_ok=True)

    # # Plot 1: Re-identification accuracy + utility loss (right axis)
    # fig1, ax1 = plt.subplots(figsize=(5.5, 4))
    # ax1b = ax1.twinx()
    # for i, (dataset, acc_vals, ul_vals) in enumerate(zip(datasets, accuracy_list, utility_loss_list)):
    #     c = colors[i % len(colors)]
    #     ax1.plot(x, [acc/baseline_accuracy[dataset] for acc in acc_vals], marker="o", color=c, linestyle="-", label=dataset)
    #     ax1b.plot(x, ul_vals, marker="s", color=c, linestyle="--", alpha=0.5)
    # ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(eps_labels, fontsize=10, rotation=45, ha="right")
    # ax1.set_xlabel("Epsilon", fontsize=13)
    # ax1.set_ylabel("RIR Gain (%)", fontsize=12)
    # ax1b.set_ylabel("Utility Loss", fontsize=12)
    # ax1.legend(loc="best", fontsize=10, framealpha=0.7)
    # fig1.tight_layout()
    # fig1.savefig(f"paper_figures/robustness_{eval_type}_reidentification.png", dpi=150, bbox_inches="tight")
    # print(f"Saved paper_figures/robustness_{eval_type}_reidentification.png")

    # # Plot 2: Mean PCC + utility loss (right axis)
    # fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    # ax2b = ax2.twinx()
    # for i, (dataset, pcc_vals, ul_vals) in enumerate(zip(datasets, pcc_list, utility_loss_list)):
    #     c = colors[i % len(colors)]
    #     ax2.plot(x, pcc_vals, marker="o", color=c, linestyle="-", label=dataset)
    #     ax2b.plot(x, ul_vals, marker="s", color=c, linestyle="--", alpha=0.5)
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(eps_labels, fontsize=10, rotation=45, ha="right")
    # ax2.set_xlabel("Epsilon", fontsize=13)
    # ax2.set_ylabel("Mean PCC", fontsize=13)
    # ax2b.set_ylabel("Utility Loss", fontsize=12)
    # ax2.legend(loc="best", fontsize=10, framealpha=0.7)
    # fig2.tight_layout()
    # fig2.savefig(f"paper_figures/robustness_{eval_type}_reconstruction.png", dpi=150, bbox_inches="tight")
    # print(f"Saved paper_figures/robustness_{eval_type}_reconstruction.png")

    ## Plot the feature preservation for ECG datasets
    # dataset_paths = {
    #     "arrhythmia": "src/results/ipopt/arrhythmia_xl/",
    #     "ptbxl":      "src/results/ipopt/ptbxl/",
    #     "ltdb": "src/results/ipopt/ltdb"
    # }
    dataset_paths = {
        "tdbrain": "src/results/ipopt/eeg_ind_500/"
    }
    plot_feature_relative_error_boxplot(dataset_paths, category="eeg", fs=100, out="paper_figures/feature_relative_error_boxplot_eeg.png")