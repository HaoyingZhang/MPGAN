import stumpy, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter, MaxNLocator
import json, sys, os
from collections import Counter
from scipy.spatial.distance import euclidean

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # Add root directory to path
from src.utils import rmse, pearson_correlation
from eval import conformal_prediction
from eval import compute_loss_from_folder, compute_utility_loss_from_folder
from reidentify import reidentification_attack
from features_extraction import extract_ecg_features, extract_eeg_features, extract_eeg_features_bis
import argparse

# Times-like serif text and math (STIX), as in most papers; fonts embedded as TrueType in pdf/ps outputs
matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                            "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})

# RIR of each dataset, used for the 5% threshold of the RIR gain plots
RIR = {"ptbxl": 0.125, "ptb_diagnostic": 0.09, "arrhythmia": 0.238, "ltdb": 0.492}


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

    # --- style ---
    fs_title, fs_legend, fs_tick = 18, 18, 18

    # --- layout ---
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.12)

    # top row narrower than the figure, centred
    gs_top = gs[0, :].subgridspec(1, 3, width_ratios=[1, 8, 1])
    ax_ts  = fig.add_subplot(gs_top[0, 1])
    ax_mpd = fig.add_subplot(gs[1, 0])   # bottom-left  : MP distance
    ax_mpi = fig.add_subplot(gs[1, 1])   # bottom-right : MP index

    # legends below each axis so that they never hide the curves
    legend_kw = dict(fontsize=fs_legend, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)

    # --- top: time series ---
    ax_ts.plot(ts_deepmp, color="blue",  label=f"DeepMP (PCC = {pcc_ts_blue:.2f})", linewidth=1.2)
    if ts_intermediate is not None:
        ax_ts.plot(ts_intermediate, color="red", linestyle="dotted", label="IPOPT Intermediate Solution", linewidth=1.0)
    ax_ts.plot(ts_ipopt,  color="red",   label=f"DeepMP+IPOPT (PCC = {pcc_ts_red:.2f})", linewidth=1.2)
    ax_ts.plot(ts_original, color="green",  label="Ground Truth", linewidth=1.2)
    ax_ts.set_title("Time Series", fontsize=fs_title)
    ax_ts.legend(ncol=2 if ts_intermediate is not None else 3, **legend_kw)

    # --- bottom-left: MP distance ---
    ax_mpd.plot(mp_blue[:, 0], color="blue", label=f"MPD DeepMP (RMSE = {rmse_mpd_blue:.2f})", linewidth=1.2)
    ax_mpd.plot(mp_red[:, 0],  color="red",  label=f"MPD DeepMP+IPOPT (RMSE = {rmse_mpd_red:.2f})", linewidth=1.2)
    ax_mpd.plot(mp_gray[:, 0], color="green", label="MPD Ground Truth", linewidth=1.2)
    ax_mpd.set_title("Matrix Profile Distance (MPD)", fontsize=fs_title)
    ax_mpd.legend(ncol=1, **legend_kw)

    # --- bottom-right: MP index ---
    x_blue = np.arange(len(mp_blue))
    x_red  = np.arange(len(mp_red))
    x_gray = np.arange(len(mp_gray))

    ax_mpi.scatter(x_blue, mp_blue[:, 1], color="blue", label=f"MPI DeepMP (Accuracy = {distance_mpi_blue:.2f})", s=4)
    ax_mpi.scatter(x_red,  mp_red[:, 1],  color="red",  label=f"MPI DeepMP+IPOPT (Accuracy = {distance_mpi_red:.2f})", s=4)
    ax_mpi.scatter(x_gray, mp_gray[:, 1], color="green", label="MPI Ground Truth", s=4
                   , marker="."
                   )
    ax_mpi.set_title("Matrix Profile Index (MPI)", fontsize=fs_title)
    ax_mpi.legend(ncol=1, markerscale=3, **legend_kw)

    for ax in (ax_ts, ax_mpd, ax_mpi):
        ax.tick_params(labelsize=fs_tick)

    return fig

def plot_rank_distribution(
    rank_map,
    save_path,
    title="Rank Distribution for person 1",
    threshold=0.75
):
    fs_text = 16
    fig, ax = plt.subplots(figsize=(5, 3))

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
                fontsize=fs_text,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )
        else:
            ax.annotate(
                f"({x_cross}, {y_cross:.2f})",
                xy=(x_cross, y_cross),
                xytext=(x_cross - max(sorted_ranks) * 0.03, y_cross + 0.1),
                fontsize=fs_text,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

    ax.set_xlabel("Rank", fontsize=fs_text)
    ax.set_ylabel("Cumulative RIR", fontsize=fs_text)
    # ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=15)
    ax.legend(loc="best", fontsize=fs_text)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved in {save_path}")
    return


def plot_feature_relative_error_boxplot(
    dataset_paths,
    category="ecg",
    fs=100,
    out="paper_figures/feature_relative_error_boxplot.png",
    features_per_row=7,
):
    """
    For each dataset in dataset_paths, load every {category}_{N}/results.json,
    compare features extracted from the reference signal ("time_series" key) and
    the IPOPT reconstruction ("solutions"[0]), compute the relative error per
    feature, and display side-by-side box plots — one colour per dataset.
    Features are split across multiple rows with `features_per_row` per row.

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
    features_per_row : int
        Number of features displayed per subplot row.
    """
    from matplotlib.lines import Line2D
    import matplotlib.ticker as ticker
    import math

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
            rec_raw = np.array(data["solutions"][0], dtype=np.float64)
            # rec_raw = np.array(data["smoothed"], dtype=np.float64)

            try:
                if category == "ecg":
                    f_ref = extract_ecg_features(ref_raw, fs=fs)
                    f_rec = extract_ecg_features(rec_raw, fs=fs)
                else:
                    f_ref = extract_eeg_features_bis(ref_raw, fs=fs)
                    f_rec = extract_eeg_features_bis(rec_raw, fs=fs)
                
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
    # 2. Build the figure — one row per chunk of `features_per_row` features
    # ------------------------------------------------------------------ #
    matplotlib.rcParams.update({"font.size": 18})

    box_w   = 0.22
    group_w = n_datasets * box_w + 0.18

    n_feats  = len(feature_keys)
    n_rows   = math.ceil(n_feats / features_per_row)
    row_w    = features_per_row * (0.35 + 0.12 * n_datasets)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(10, row_w), 6.5 * n_rows),
        squeeze=False,
    )

    legend_handles = [
        Line2D([0], [0], marker=markers[i], color="none",
               markerfacecolor=colors[i], markersize=8,
               label=ds, markeredgewidth=0)
        for i, ds in enumerate(ds_list)
    ]
    ncol = min(n_datasets, 4)

    for row_idx in range(n_rows):
        ax = axes[row_idx, 0]
        chunk = feature_keys[row_idx * features_per_row : (row_idx + 1) * features_per_row]
        n_chunk = len(chunk)

        for di, (dataset, errors) in enumerate(all_errors.items()):
            color         = colors[di]
            marker        = markers[di]
            positions     = [i * group_w + di * box_w for i in range(n_chunk)]
            data_per_feat = [errors.get(k, [np.nan]) for k in chunk]

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
                        fontsize=10, color="black", fontweight="bold")

        tick_pos = [i * group_w + (n_datasets - 1) * box_w / 2 for i in range(n_chunk)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(chunk, rotation=45, ha="right", fontsize=18)
        ax.set_ylabel("Relative error", fontsize=20)
        ax.set_yscale("log")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.tick_params(axis="y", labelsize=17)

        if row_idx == 0:
            ax.legend(handles=legend_handles, fontsize=16, loc="upper left",
                      framealpha=0.85, edgecolor="#cccccc", ncol=ncol)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="ecg", help="Dataset category")
    parser.add_argument("--exp_name",type=str, choices=["robustness", "plot_ipopt", "conformal_prediction", "features"], help="Use solution after ipopt")
    parser.add_argument("--datasets", nargs="+", default=["ptbxl"], help="Datasets plotted in the robustness experiment")
    parser.add_argument("--rob", nargs="+", default=["mpd", "mpi"], choices=["mpd", "mpi", "both"],
                        help="Perturbation modes plotted in the robustness experiment")

    args = parser.parse_args()

    if args.exp_name == "plot_ipopt":
        if args.category == "ecg":
            deepmp_sol_path = "src/results/baseline/ptbxl/ptbxl/ecg_39/results.json"
            ipopt_only_path = "src/results/ipopt/ptbxl_with_only_ipopt_sol/ecg_39/results.json"
            ipopt_sol_path = "src/results/ipopt/ptbxl/ecg_39/results.json"
        else:
            # deepmp_sol_path = "src/results/baseline/eeg/tdbrain/eeg_70/results.json"
            # ipopt_sol_path = "src/results/ipopt/eeg/eeg_70/results.json"
            deepmp_sol_path = "src/results/baseline/eeg_160hz/tdbrain/eeg_170/results.json"
            ipopt_sol_path = "src/results/ipopt/eeg_160hz/eeg_170/results.json"

        with open(deepmp_sol_path, "r") as f:
            res = json.load(f)
            ts_deepmp = res["fake_data"]
            ts_original = res["data"]
            
        with open(ipopt_sol_path, "r") as f:
            res = json.load(f)
            ts_ipopt = res["solutions"][0]
            ts_smooth = res["smoothed"]
        
        if args.category == "ecg":
            with open(ipopt_only_path, "r") as f:
                res = json.load(f)
                ts_ipopt_only = res["solutions"][0]
        
        if args.category == "ecg":
            fig = plot_three_ts_with_mp(ts_deepmp, ts_ipopt, ts_original, None, m=100, znorm=True)
            save_path = "paper_figures/compare_ipopt_ecg.jpg"
        else:
            fig = plot_three_ts_with_mp(ts_deepmp, ts_smooth, ts_original, ts_ipopt, m=100, znorm=True)
            save_path = "paper_figures/compare_ipopt_eeg.jpg"
        
        print(f"Figure saved in {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if args.exp_name == "conformal_prediction":
        # Conformal prediction
        threshold = {"arrhythmia":0.7, "ptbxl":0.66, "ptb_diagnostic":0.66, "tdbrain":0.22}
        for dataset in ["tdbrain"]:
            base_test_folder =  f"src/results/ipopt/eeg/" if args.category == "eeg" else f"src/results/ipopt/{dataset}/"
            ranks = conformal_prediction(n=500, m=100, base_folder_test=base_test_folder, ref_index=[1000,1500,2000], using_features=True, dataset=dataset, metric=euclidean)
            ranks_mp = conformal_prediction(n=500, m=100, base_folder_test=base_test_folder, ref_index=[0], using_features=False, dataset=dataset, metric=euclidean, using_mp=True)
            plot_rank_distribution({"TS reconstructed": ranks, "MP": ranks_mp}, f"paper_figures/rank_distribution_{dataset}.png", title="Rank Distribution", threshold=threshold[dataset])


    # Robustness
    if args.exp_name == "robustness":
        # Plots the raw results stored by test/eval_robustness.py (see exe_robustness.sh) in
        # <root>/<dataset>/<rob>/<eps dir>/eval_metrics.json: per-sample metrics as
        # [[run 1 samples], ...] and the re-identification rate as [run 1, ...].
        # Each run is first averaged over its samples, then mean and standard error over runs.
        eval_types = args.rob
        # same sizes as the compare_ipopt figures
        fs_legend = 18
        fs_text = 18
        fs_label = 15  # tick labels
        vec_size = 401
        datasets = args.datasets
        root_path = "src/results/baseline/ptbxl/" if args.category == "ecg" else "src/results/baseline/eeg_160hz/"

        def run_stats(runs):
            per_run = [np.nanmean(r) if isinstance(r, list) else r for r in runs]
            if not per_run:
                return np.nan, np.nan
            return np.nanmean(per_run), np.nanstd(per_run) / np.sqrt(len(per_run))

        rir_limit = 0.05

        for eval_type in eval_types:
            epsilon = [100.0,1000.0,10000.0,100000.0,500000.0,800000.0, 1000000.0,100000000.0]

            # older runs (before eval_metrics.json) were cached per run as {dataset: {eps: {"pcc", "acc",
            # "utility_loss"}}}, one file per utility metric: the 1-PCC of the MPD for mpd, the EMD
            # otherwise, and the RMSE of mpi in a separate file
            legacy_files = {"pcc" if eval_type == "mpd" else "emd": f"eval_robustness_{eval_type}_results.json"}
            if eval_type == "mpi":
                legacy_files["rmse"] = f"eval_robustness_{eval_type}_results_rmse.json"
            legacy = {}
            for utility_name, file_name in legacy_files.items():
                legacy_file = os.path.join(root_path, file_name)
                if os.path.exists(legacy_file):
                    with open(legacy_file) as f:
                        legacy[utility_name] = json.load(f)

            # raw results, per dataset and eps
            results = {}
            for dataset in datasets:
                results[dataset] = []
                for eps in epsilon:
                    eps_dir = f"{eval_type}{eps}" if eval_type in ("mpd", "mpi") else str(eps)
                    metrics_file = os.path.join(root_path, dataset, eval_type, eps_dir, "eval_metrics.json")
                    res = {}
                    if os.path.exists(metrics_file):
                        with open(metrics_file) as f:
                            res = json.load(f)
                    else:
                        # each legacy file comes from its own runs: the reconstruction PCC and the
                        # re-identification rate are kept per utility metric, {utility metric: runs}
                        for utility_name, old_results in legacy.items():
                            old = old_results.get(dataset, {}).get(str(eps))
                            if old is None:
                                continue
                            res.setdefault("reconstruction_pcc", {})[utility_name] = old["pcc"]
                            res.setdefault("reidentification_rate", {})[utility_name] = old["acc"]
                            res[utility_name] = old["utility_loss"]
                            print(f"[{dataset}] {eval_type} eps={eps}: using the older results of "
                                  f"{legacy_files[utility_name]}")
                        # the RMSE file comes from other runs: plot the same RIR as with the EMD
                        rir_runs = res.get("reidentification_rate", {})
                        if "emd" in rir_runs and "rmse" in rir_runs:
                            rir_runs["rmse"] = rir_runs["emd"]
                        if not res:
                            print(f"[{dataset}] missing {metrics_file}, run exe_robustness.sh first")
                    results[dataset].append(res)

            def metric_stats(name, utility_name=None):
                """[mean per eps] and [standard error per eps] of one metric, one list per dataset.
                utility_name picks the runs of a privacy metric stored per utility metric (legacy results)."""
                def runs_of(res):
                    runs = res.get(name, [])
                    return runs.get(utility_name, []) if isinstance(runs, dict) else runs

                means, stds = [], []
                for dataset in datasets:
                    stats_eps = [run_stats(runs_of(res)) for res in results[dataset]]
                    means.append([s[0] for s in stats_eps])
                    stds.append([s[1] for s in stats_eps])
                return means, stds

            # every other stored metric is a utility metric (MP distances and peak metrics)
            utility_names = sorted({name for dataset in datasets for res in results[dataset] for name in res}
                                   - {"reconstruction_pcc", "reidentification_rate"})
            utility_labels = {"pcc": "1-PCC (MPD)", "emd": "EMD (MPI)", "rmse": "RMSE (MPI)"}

            def fmt_eps(v):
                exp = int(np.floor(np.log10(v)))
                coeff = round(v / 10**exp, 1)
                return f"${coeff}\\times10^{{{exp}}}$"

            eps_labels = [fmt_eps(e/vec_size) for e in epsilon]
            eps_xlabels = {"mpd": "$\\epsilon_D$", "mpi": "$\\epsilon_I$"}

            matplotlib.rcParams.update({"font.size": fs_text})
            x = np.arange(len(epsilon))
            # same style as test/eval_utility.py: utility in blue on the left axis,
            # privacy metric in orange dotted on the right axis
            c_util, c_priv = "tab:blue", "tab:orange"
            # datasets told apart by their marker
            util_markers = ["o", "^", "v", "D"]
            priv_markers = ["s", "P", "X", "*"]
            # one folder per dataset (or per combination when several are plotted together)
            out_dir = os.path.join("paper_figures/robustness", "_".join(datasets))
            os.makedirs(out_dir, exist_ok=True)

            # reference eps of the gain plots: the privacy metric is divided by its value at this eps
            ref_eps_idx = epsilon.index(100000000.0)

            def plot_figure(scale, priv_name, priv_label, utility_name, out, gain=False):
                priv_vals, priv_stds = metric_stats(priv_name, utility_name)
                ul_vals, ul_stds = metric_stats(utility_name)
                if gain:
                    priv_stds = [np.array(s) * scale/ v[ref_eps_idx] for v, s in zip(priv_vals, priv_stds)]
                    priv_vals = [np.array(v) * scale / v[ref_eps_idx] for v in priv_vals]
                util_label = utility_labels.get(utility_name, utility_name)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                axb = ax.twinx()
                for i, dataset in enumerate(datasets):
                    prefix = f"{dataset} " if len(datasets) > 1 else ""
                    ax.errorbar(x, ul_vals[i], yerr=ul_stds[i],
                                marker=util_markers[i % len(util_markers)], color=c_util, linestyle="-",
                                label=prefix + util_label, capsize=4, capthick=1.2, elinewidth=1.2)
                    axb.errorbar(x, priv_vals[i], yerr=priv_stds[i],
                                 marker=priv_markers[i % len(priv_markers)], color=c_priv, linestyle=":",
                                 label=prefix + priv_label, capsize=4, capthick=1.2, elinewidth=1.2)

                if priv_name == "reidentification_rate":
                    # 5% RIR
                    if not gain or any(dataset in RIR for dataset in datasets):
                        axb.axhline(y=rir_limit, color="gray", linestyle=":", linewidth=1.2)
                    axb.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                else:
                    axb.set_ylim(0, 0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(eps_labels, fontsize=fs_label, rotation=45, ha="right")
                ax.set_xlabel(eps_xlabels.get(eval_type, "Epsilon"), fontsize=fs_text)
                ax.set_ylabel(util_label, fontsize=fs_text, color=c_util)
                axb.set_ylabel(priv_label, fontsize=fs_text, color=c_priv)
                ax.tick_params(axis="y", colors=c_util, labelsize=fs_label)
                axb.tick_params(axis="y", colors=c_priv, labelsize=fs_label)
                # denser y ticks on both axes (the x axis has one tick per eps already)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
                axb.yaxis.set_major_locator(MaxNLocator(nbins=8))
                lines = ax.get_legend_handles_labels()
                lines_b = axb.get_legend_handles_labels()
                fig.tight_layout()
                # legend below the figure (under the x label) so that it never hides the curves
                # fig.legend(lines[0] + lines_b[0], lines[1] + lines_b[1], loc="upper center",
                #            bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=fs_legend, frameon=False)
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {out}")

            for utility_name in utility_names:
                utility_tag = utility_name.replace("@", "_at")
                # plot_figure("reidentification_rate", "RIR", utility_name,
                #             os.path.join(out_dir, f"{eval_type}_reidentification_{utility_tag}.png"))
                plot_figure(RIR[dataset], "reidentification_rate", "RIR", utility_name,
                            os.path.join(out_dir, f"{eval_type}_reidentification_{utility_tag}.png"), gain=True)
                plot_figure(1, "reconstruction_pcc", "Mean PCC", utility_name,
                            os.path.join(out_dir, f"{eval_type}_reconstruction_{utility_tag}.png"))


    if args.exp_name == "features":
        ## Plot the feature preservation for ECG datasets
        if args.category == "ecg":
            dataset_paths = {
                "arrhythmia": "src/results/ipopt/arrhythmia_xl/",
                "ptbxl":      "src/results/ipopt/ptbxl/",
                "ltdb": "src/results/ipopt/ltdb",
                "ptb_diagnostic": "src/results/ipopt/ptb_diagnostic"
            }
            fs = 100
        else:
            dataset_paths = {
                "eegmmidb": "src/results/ipopt/eegmmidb/",
                "tdbrain": "src/results/ipopt/eeg_160hz/"
            }
            fs = 160
        plot_feature_relative_error_boxplot(dataset_paths, category=args.category, fs=fs, out=f"paper_figures/feature_relative_error_boxplot_{args.category}.png", features_per_row=17)