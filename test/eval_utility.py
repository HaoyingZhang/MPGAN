"""
Utility vs. re-identification when only the MPI budget varies.

For every eps_mpi in --eps_mpi (MPD budget fixed to a large --eps_mpd, i.e. almost no
MPD noise), each run:
  1. calls test/eval_robustness.py --rob both --eps_mpi eps_mpi --eps_mpd eps_mpd,
  2. scores every released MP against the original one with the peak metrics of
     src/utils.py (motif recall / pair accuracy / top-1 hit, discord recall),
  3. runs the re-identification attack on the reconstructions.
Runs are cached in --results so an interrupted sweep resumes where it stopped.

One figure per utility metric: the metric on the left y axis, the re-identification
rate on the right y axis, eps_mpi (per MP element) on the x axis.

Usage (from the repository root):
    python3 test/eval_utility.py
    python3 test/eval_utility.py --eps_mpd 1e8 --eps_mpi 100 1000 1e4 1e5 1e6 --runs 5
    python3 test/eval_utility.py --metrics motif_pair_acc@3 top1_motif_hit
    python3 test/eval_utility.py --reuse --key reconstruct --eps_mpi 10000 100000 500000 --runs 1
    python3 test/eval_utility.py --ipopt --ipopt_time_limit 120
"""
import argparse, json, os, re, subprocess, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)  # Add root directory to path

from src.utils import mp_utility_metrics_motif, mp_utility_metrics_discord
from reidentify import reidentification_attack

# Times-like serif text and math (STIX), as in most papers; fonts embedded as TrueType in pdf/ps outputs
matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                            "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})

RESULTS_ROOT = {"ecg": "src/results/baseline/ptbxl/", "eeg": "src/results/baseline/eeg_160hz/"}
# re-identification rate without perturbation, per dataset; the reported rate is rescaled so that
# the eps_mpi=1e8 setting (practically no noise) matches it
MAX_RIR = {"ptbxl": 0.125, "arrhythmia": 0.238, "ptb_diagnostic": 0.09, "ltdb": 0.49,
           "eegmmidb": 0.1, "tdbrain": 0.05}
RIR_REF_EPS = 100000000.0
# the reported re-identification rate is the max of the attack on all the features and on these ones only
# (ECG features of features_extraction.extract_ecg_features)
FEATURE_SUBSETS = {"ecg": [["bpm", "arrhythmia"]], "eeg": []}


def fmt_metric(name):
    """Plot label of a metric: motif_recall@3 -> motif_recall (k=3)."""
    return re.sub(r"@(\d+)$", r" (k=\1)", name)


def fmt_eps(v):
    exp = int(np.floor(np.log10(v)))
    coeff = round(v / 10**exp, 1)
    return f"${coeff}\\times10^{{{exp}}}$"


def utility_from_folder(folder, category, m, k, record=True):
    """Mean of every peak metric over the <category>_i/results.json of one run (NaNs ignored).
    With record, each sample's metrics are also appended to its results.json under "utility_metrics"
    ({metric: [run 1, run 2, ...]}); the MPs themselves are overwritten by every run."""
    values = {}
    for name in os.listdir(folder):
        if not re.fullmatch(rf"{category}_\d+", name):
            continue
        res_file = os.path.join(folder, name, "results.json")
        with open(res_file) as f:
            d = json.load(f)
        mp_orig, mp_pert = np.asarray(d["mp_original"]), np.asarray(d["mp_perturbed"])
        P, I = mp_orig[:, 0], mp_orig[:, 1].astype(int)
        P_rel, I_rel = mp_pert[:, 0], mp_pert[:, 1].astype(int)
        metrics = {**mp_utility_metrics_motif(P, I, P_rel, I_rel, m, k=k),
                   **mp_utility_metrics_discord(P, P_rel, m, k=k)}
        if record:
            history = d.setdefault("utility_metrics", {})
            for metric, v in metrics.items():
                past = history.get(metric, [])
                history[metric] = (past if isinstance(past, list) else [past]) + [float(v)]
            with open(res_file, "w") as f:
                json.dump(d, f, indent=4)
        for metric, v in metrics.items():
            values.setdefault(metric, []).append(v)
    return {metric: float(np.nanmean(v)) for metric, v in values.items()}


def run_setting(args, eps_mpi):
    """One eval_robustness run (skipped with --reuse, which keeps the existing folder and its
    "reconstruct" key) + its utility metrics and re-identification accuracy."""
    if not args.reuse:
        cmd = ["python3", "test/eval_robustness.py", "-dataset", args.dataset,
               "--rob", "both", "--eps_mpi", str(eps_mpi), "--eps_mpd", str(args.eps_mpd),
               "-n_ts", str(args.n_ts), "-n", str(args.n), "-m", str(args.m), "-c", args.category,
               "-g_model", "deepmp", "-mp_embedding", "-znorm", "-fill", "100"]
        if args.ipopt:
            # IPOPT reconstruction from the perturbed MP, stored under "reconstruct" in each results.json
            cmd += ["--ipopt", "--ipopt_time_limit", str(args.ipopt_time_limit)]
        subprocess.run(cmd, check=True)
    folder = os.path.join(RESULTS_ROOT[args.category], args.dataset, "both", f"mpi{eps_mpi}_mpd{args.eps_mpd}")
    res = utility_from_folder(folder, args.category, args.m, args.k, record=not args.reuse)
    res["acc"] = max(float(reidentification_attack(folder, ipopt=False, dataset=args.dataset, category=args.category,
                                                   use_mp=False, classifier="svm", use_feature=True,
                                                   use_original=False, distance=False, ts_key=args.key,
                                                   feature_subset=subset))
                     for subset in [None] + FEATURE_SUBSETS[args.category])
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl")
    ap.add_argument("--category", default="ecg", choices=list(RESULTS_ROOT))
    ap.add_argument("--eps_mpi", type=float, nargs="+",
                    default=[100.0, 1000.0, 10000.0, 100000.0, 500000.0, 800000.0, 1000000.0, 100000000.0])
    ap.add_argument("--eps_mpd", type=float, default=100000.0, help="fixed (large) MPD budget")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--n_ts", type=int, default=200, help="number of people evaluated (the first n_ts)")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("-m", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="number of peaks for the @k metrics")
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="utility metrics to plot (default: all of them, one figure each)")
    ap.add_argument("--combined", action="store_true",
                    help="plot all the utility metrics in one figure together with the re-identification rate")
    ap.add_argument("--reuse", action="store_true",
                    help="do not rerun eval_robustness (which overwrites results.json and drops \"reconstruct\"); "
                         "evaluate the existing folders")
    ap.add_argument("--key", default=None,
                    help="results.json key attacked by the re-identification, e.g. reconstruct (default: fake_data)")
    ap.add_argument("--ipopt", action="store_true",
                    help="reconstruct with IPOPT in each eval_robustness run and attack the reconstructions (--key reconstruct)")
    ap.add_argument("--ipopt_time_limit", type=float, default=120.0, help="IPOPT time limit per time series in seconds")
    ap.add_argument("--results", default=None, help="cache of per-run results (json)")
    ap.add_argument("--out_dir", default="paper_figures")
    args = ap.parse_args()
    if args.ipopt and args.key is None:
        args.key = "reconstruct"

    key_suffix = f"_{args.key}" if args.key else ""
    results_path = args.results or os.path.join(RESULTS_ROOT[args.category],
                                                f"eval_utility_mpi_mpd{args.eps_mpd}_k{args.k}{key_suffix}_results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    runs_ds = all_results.setdefault(args.dataset, {})

    for eps_mpi in args.eps_mpi:
        runs = runs_ds.setdefault(str(eps_mpi), [])
        for run in range(len(runs), args.runs):
            res = run_setting(args, eps_mpi)
            runs.append(res)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"[{args.dataset}] eps_mpi={eps_mpi} eps_mpd={args.eps_mpd} run {run + 1}/{args.runs}: "
                  + " ".join(f"{name}={v:.3f}" for name, v in res.items()))

    # mean and standard error over runs, per setting
    def stat(name):
        vals = [[r[name] for r in runs_ds[str(e)]] for e in args.eps_mpi]
        return (np.array([np.nanmean(v) for v in vals]),
                np.array([np.nanstd(v) / np.sqrt(max(len(v), 1)) for v in vals]))

    utility_names = [name for name in runs_ds[str(args.eps_mpi[0])][0] if name != "acc"]
    metrics = args.metrics or utility_names
    acc_mean, acc_sem = stat("acc")
    acc_ref = np.nanmean([r["acc"] for r in runs_ds[str(RIR_REF_EPS)]])
    acc_scale = MAX_RIR[args.dataset] / acc_ref
    acc_mean, acc_sem = acc_mean * acc_scale, acc_sem * acc_scale

    vec_size = args.n - args.m + 1
    x = np.arange(len(args.eps_mpi))
    eps_labels = [fmt_eps(e / vec_size) for e in args.eps_mpi]
    matplotlib.rcParams.update({"font.size": 13})
    c_acc = "tab:orange"
    # one color per utility metric (default cycle without the orange of the re-identification),
    # the same in the combined figure and in the per-metric figures
    util_colors = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"] if c != "#ff7f0e"]
    c_utils = {metric: util_colors[i % len(util_colors)] for i, metric in enumerate(metrics)}
    os.makedirs(args.out_dir, exist_ok=True)

    if args.combined:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        axb = ax.twinx()
        markers = ["o", "^", "v", "D", "P", "X"]
        for i, metric in enumerate(metrics):
            util_mean, util_sem = stat(metric)
            ax.errorbar(x, util_mean, yerr=util_sem, marker=markers[i % len(markers)],
                        color=c_utils[metric], linestyle="-",
                        label=fmt_metric(metric), capsize=4, capthick=1.2, elinewidth=1.2)
        axb.errorbar(x, acc_mean, yerr=acc_sem, marker="s", color=c_acc, linestyle=":",
                     label="Re-identification", capsize=4, capthick=1.2, elinewidth=1.2)
        for xi, a in zip(x, acc_mean):
            axb.annotate(f"{a * 100:.1f}%", (xi, a), textcoords="offset points", xytext=(0, 9),
                         ha="center", va="bottom", fontsize=10, color=c_acc)
        ax.set_ylim(-0.05, 1.05)
        axb.set_ylim(0, 0.2)
        axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}"))
        ax.set_xticks(x)
        ax.set_xticklabels(eps_labels, fontsize=12, rotation=45, ha="right")
        ax.set_xlabel(f"$\\epsilon_{{I}}$ ($\\epsilon_{{D}}$ = {fmt_eps(args.eps_mpd / vec_size)})", fontsize=16)
        ax.set_ylabel("Utility", fontsize=16)
        axb.set_ylabel("Re-identification rate (%)", fontsize=14, color=c_acc)
        axb.tick_params(axis="y", colors=c_acc)
        lines = ax.get_legend_handles_labels()
        lines_b = axb.get_legend_handles_labels()
        # above the axes so that it never hides the curves
        ax.legend(lines[0] + lines_b[0], lines[1] + lines_b[1], loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=3, fontsize=10, framealpha=0.7)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"utility_{args.dataset}_mpi_combined.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
        return

    for metric in metrics:
        util_mean, util_sem = stat(metric)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        axb = ax.twinx()
        c_util = c_utils[metric]
        ax.errorbar(x, util_mean, yerr=util_sem, marker="o", color=c_util, linestyle="-",
                    label=fmt_metric(metric), capsize=4, capthick=1.2, elinewidth=1.2)
        axb.errorbar(x, acc_mean, yerr=acc_sem, marker="s", color=c_acc, linestyle=":",
                     label="Re-identification", capsize=4, capthick=1.2, elinewidth=1.2)
        for xi, a in zip(x, acc_mean):
            axb.annotate(f"{a * 100:.1f}%", (xi, a), textcoords="offset points", xytext=(0, 9),
                         ha="center", va="bottom", fontsize=10, color=c_acc)
        ax.set_ylim(-0.05, 1.05)
        axb.set_ylim(0, 0.2)
        axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}"))
        ax.set_xticks(x)
        ax.set_xticklabels(eps_labels, fontsize=12, rotation=45, ha="right")
        ax.set_xlabel(f"$\\epsilon_{{MPI}}$ ($\\epsilon_{{MPD}}$ = {fmt_eps(args.eps_mpd / vec_size)})", fontsize=16)
        ax.set_ylabel(fmt_metric(metric), fontsize=16, color=c_util)
        axb.set_ylabel("Re-identification rate (%)",fontsize=14, color=c_acc)
        ax.tick_params(axis="y", colors=c_util)
        axb.tick_params(axis="y", colors=c_acc)
        lines = ax.get_legend_handles_labels()
        lines_b = axb.get_legend_handles_labels()
        ax.legend(lines[0] + lines_b[0], lines[1] + lines_b[1], loc="center left", fontsize=12, framealpha=0.7)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"utility_{args.dataset}_mpi_{metric.replace('@', '_at')}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
