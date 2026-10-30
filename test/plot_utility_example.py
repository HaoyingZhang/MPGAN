"""
Step-by-step computation of the four peak utility metrics (src/utils.py) on one sample.

Reads the original and perturbed MPs of one results.json (written by test/eval_robustness.py),
prints every intermediate step (peaks, matches, pointers) and plots the original vs perturbed
MPD with the motifs / discords and the MPI with the pointers checked by motif_pair_acc.

Usage (from the repository root):
    python3 test/plot_utility_example.py \
        --results src/results/baseline/ptbxl/ptbxl/both/mpi100000.0_mpd100000.0/ecg_0/results.json \
        --out paper_figures/utility_example_ptbxl_ecg0_mpi100000.0.png
"""
import argparse, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linear_sum_assignment

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)  # Add root directory to path

from src.utils import extract_peaks, match_peaks, mp_utility_metrics_motif, mp_utility_metrics_discord


# Times-like serif text and math (STIX), as in most papers; fonts embedded as TrueType in pdf/ps outputs
matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                            "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})


# paper notation of the peak utility metrics (title)
METRIC_SYMBOLS = {"motif_recall": r"$A^{\mathrm{mot}}$", "motif_pair_acc": r"$A^{\mathrm{pair}}$",
                  "top1_motif_hit": r"$A^{\mathrm{top1}}$", "discord_recall": r"$A^{\mathrm{disc}}$"}


def fmt_metric(name):
    """Plot label of a metric: motif_recall@3 -> motif_recall (k=3)."""
    return re.sub(r"@(\d+)$", r" (k=\1)", name)


def rejected_pairs(orig_peaks, rel_peaks, delta):
    """Pairs of the same one-to-one assignment as match_peaks that are dropped (distance > delta)."""
    if len(orig_peaks) == 0 or len(rel_peaks) == 0:
        return []
    cost = np.abs(orig_peaks[:, None] - rel_peaks[None, :])
    rows, cols = linear_sum_assignment(cost)
    return [(int(orig_peaks[r]), int(rel_peaks[c])) for r, c in zip(rows, cols) if cost[r, c] > delta]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.json of one sample (with mp_original / mp_perturbed)")
    ap.add_argument("--out", default=None, help="output figure, .png or .pdf (default: no figure)")
    ap.add_argument("--dpi", type=int, default=300, help="resolution of a .png figure")
    ap.add_argument("-m", type=int, default=100, help="subsequence length")
    ap.add_argument("--k", type=int, default=3, help="number of peaks for the @k metrics")
    args = ap.parse_args()

    m, k = args.m, args.k
    excl = delta = m // 2  # defaults of src/utils.py

    with open(args.results) as f:
        d = json.load(f)
    mp_orig, mp_pert = np.asarray(d["mp_original"]), np.asarray(d["mp_perturbed"])
    P, I = mp_orig[:, 0], mp_orig[:, 1].astype(int)
    P_rel, I_rel = mp_pert[:, 0], mp_pert[:, 1].astype(int)
    print(f"L = {len(P)}, m = {m}, k = {k}, excl = delta = {excl}")
    print(f"MPD changed: max |P_rel - P| = {np.max(np.abs(P_rel - P)):.4f}")
    print(f"MPI changed at {np.sum(I_rel != I)}/{len(I)} positions")

    # motifs: top-k minima of the MPD
    print("\n--- motifs (top-k minima of the MPD) ---")
    mo, mr = extract_peaks(P, k, m, excl, "min"), extract_peaks(P_rel, k, m, excl, "min")
    print("original motifs  :", [(int(i), round(float(P[i]), 3)) for i in mo])
    print("perturbed motifs :", [(int(i), round(float(P_rel[i]), 3)) for i in mr])
    matches = match_peaks(mo, mr, delta)
    rejected = rejected_pairs(mo, mr, delta)
    print("matched (|dist| <= delta):", matches)
    print("assigned but too far (|dist| > delta):", rejected)
    print(f"motif_recall@{k} = {len(matches)}/{len(mo)} = {len(matches) / len(mo):.3f}")

    # pointers of the matched motifs
    print("\n--- pointers of the matched motifs ---")
    ok = []
    for i, _ in matches:
        good = abs(I_rel[i] - I[i]) <= delta
        ok.append(good)
        print(f"i = {i}: I[i] = {I[i]}, I_rel[i] = {I_rel[i]}, |diff| = {abs(I_rel[i] - I[i])} -> {'ok' if good else 'wrong'}")
    print(f"motif_pair_acc@{k} = {sum(ok)}/{len(ok)} = {np.mean(ok):.3f}" if ok else f"motif_pair_acc@{k} = NaN")

    # best motif pair, in either order
    print("\n--- top-1 motif pair ---")
    a, b = mo[0], I[mo[0]]
    c, dd = mr[0], I_rel[mr[0]]
    same = abs(c - a) <= delta and abs(dd - b) <= delta
    swapped = abs(c - b) <= delta and abs(dd - a) <= delta
    print(f"original pair (a, I[a]) = ({a}, {b}), perturbed pair (c, I_rel[c]) = ({c}, {dd})")
    print(f"same order: {same}, swapped: {swapped} -> top1_motif_hit = {float(same or swapped)}")

    # discords: top-k maxima of the MPD
    print("\n--- discords (top-k maxima of the MPD) ---")
    do, dr = extract_peaks(P, k, m, excl, "max"), extract_peaks(P_rel, k, m, excl, "max")
    print("original discords :", [(int(i), round(float(P[i]), 3)) for i in do])
    print("perturbed discords:", [(int(i), round(float(P_rel[i]), 3)) for i in dr])
    dmatches = match_peaks(do, dr, delta)
    drejected = rejected_pairs(do, dr, delta)
    print("matched:", dmatches)
    print("assigned but too far:", drejected)
    print(f"discord_recall@{k} = {len(dmatches)}/{len(do)} = {len(dmatches) / len(do):.3f}")

    # same values with the library functions
    metrics = {**mp_utility_metrics_motif(P, I, P_rel, I_rel, m, k=k), **mp_utility_metrics_discord(P, P_rel, m, k=k)}
    print("\n--- check with the library functions ---")
    print(metrics)

    if args.out is None:
        return
    c_orig, c_rel = "tab:blue", "tab:red"
    pos = np.arange(len(P))
    # MPD and MPI on top, one zoom on the MPI per pointer checked by motif_pair_acc below
    n_zoom = max(len(matches), 1)
    # compact figure with large fonts (paper)
    fs_tick, fs_label, fs_text = 18, 20, 16
    matplotlib.rcParams.update({"font.size": fs_tick})
    fig = plt.figure(figsize=(11, 11))
    # row 2 is an empty spacer leaving room for the arrows between the MPI panel and the zooms
    gs = gridspec.GridSpec(4, n_zoom, height_ratios=[1.35, 1, 0.33, 0.95], figure=fig,
                           left=0.1, right=0.99, top=0.94, bottom=0.075, hspace=0.05, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax1.tick_params(labelbottom=False)  # same positions as the MPI panel below
    ax1.plot(pos, P, color=c_orig, label="original MPD")
    ax1.plot(pos, P_rel, color=c_rel, alpha=0.7, label="perturbed MPD")
    ax1.scatter(mo, P[mo], marker="v", s=90, color=c_orig, zorder=3, label="original motifs")
    ax1.scatter(mr, P_rel[mr], marker="v", s=90, facecolors="none", edgecolors=c_rel, linewidths=2, zorder=3,
                label="perturbed motifs")
    ax1.scatter(do, P[do], marker="^", s=90, color=c_orig, zorder=3, label="original discords")
    ax1.scatter(dr, P_rel[dr], marker="^", s=90, facecolors="none", edgecolors=c_rel, linewidths=2, zorder=3,
                label="perturbed discords")
    # assigned peak pairs with their distance vs delta: motifs below the markers, discords above
    pairs = ([(o, r, True, "min") for o, r in matches] + [(o, r, False, "min") for o, r in rejected]
             + [(o, r, True, "max") for o, r in dmatches] + [(o, r, False, "max") for o, r in drejected])
    def h_align(x):
        """Text alignment keeping labels near either end of the series inside the axes."""
        return "left" if x < 0.08 * len(P) else "right" if x > 0.92 * len(P) else "center"

    for o, r, ok, kind in pairs:
        color = "dimgray" if ok else c_rel
        ax1.annotate("", xy=(r, P_rel[r]), xytext=(o, P[o]),
                     arrowprops=dict(arrowstyle="-", color=color, ls=":" if ok else "--"))
        cmp = "$\\leq$" if ok else "$>$"
        text = f"{abs(r - o)} {cmp} $\\delta$={delta}"
        if kind == "min":
            ax1.annotate(text, ((o + r) / 2, min(P[o], P_rel[r])), textcoords="offset points", xytext=(0, -14),
                         ha=h_align((o + r) / 2), va="top", fontsize=fs_text, color=color)
        else:
            ax1.annotate(text, ((o + r) / 2, max(P[o], P_rel[r])), textcoords="offset points", xytext=(0, 12),
                         ha=h_align((o + r) / 2), va="bottom", fontsize=fs_text, color=color)
    # top-1 motif of each MP and its pointer (the pair checked by top1_motif_hit), under the distances
    # both labels on the same line; when the two top-1 are close, the perturbed one is shifted to the right
    y_top1 = min(P[mo[0]], P_rel[mr[0]])
    close = abs(int(mo[0]) - int(mr[0])) < 0.15 * len(P)
    for shift, (pos_1, ptr, color) in enumerate([(mo[0], I[mo[0]], c_orig), (mr[0], I_rel[mr[0]], c_rel)]):
        ax1.annotate(f"top-1 $\\rightarrow$ {ptr}", (pos_1, y_top1), textcoords="offset points",
                     xytext=(125 * shift if close else 0, -38), ha=h_align(pos_1), va="top",
                     fontsize=fs_text, color=color)
    names = list(metrics)
    sep = "    "
    symbol = lambda name: METRIC_SYMBOLS.get(re.sub(r"@\d+$", "", name), fmt_metric(name))
    fig.suptitle("\n".join(sep.join(f"{symbol(name)} = {metrics[name]:.2f}" for name in names[j:j + 2])
                           for j in range(0, len(names), 2)), fontsize=fs_label, y=0.995)
    ax1.set_ylabel("MPD", fontsize=fs_label)
    # legend inside, top right; headroom above the curve keeps it off the data
    lo_y, hi_y = min(P.min(), P_rel.min()), max(P.max(), P_rel.max())
    # room below for the motif distances and top-1 labels, above for the discord distances and the legend
    ax1.set_ylim(lo_y - 0.6 * (hi_y - lo_y), hi_y + 0.95 * (hi_y - lo_y))
    ax1.legend(fontsize=fs_text, ncol=3, loc="upper left", columnspacing=0.8, handletextpad=0.4, framealpha=0.9)

    ax2.scatter(pos, I, s=8, color=c_orig, label="original MPI")
    ax2.scatter(pos, I_rel, s=8, color=c_rel, alpha=0.6, label="perturbed MPI")
    for i, _ in matches:  # pointers checked by motif_pair_acc
        ax2.plot([i, i], [I[i], I_rel[i]], color="k", lw=2)
        ax2.scatter([i], [I[i]], s=60, marker="o", facecolors="none", edgecolors="k", zorder=3)
    ax2.set_xlabel("position", fontsize=fs_label)
    ax2.set_ylabel("MPI", fontsize=fs_label)
    ax2.set_ylim(-20, len(P) * 1.35)  # headroom for the legend
    ax2.legend(fontsize=fs_text, ncol=2, loc="upper right", markerscale=2.5, framealpha=0.9)

    # zoom around each matched motif i: perturbed pointer I_rel[i] vs the interval I[i] +- delta
    half = 10  # positions shown on each side
    zooms = []  # (zoom axes, bottom centre of its box on the MPI panel)
    for col, (i, _) in enumerate(sorted(matches)):  # left to right, so that the arrows do not cross
        axz = fig.add_subplot(gs[3, col])
        lo, hi = max(0, i - half), min(len(P), i + half + 1)
        # same colors as the MPI panel, whose legend applies
        axz.scatter(pos[lo:hi], I[lo:hi], s=25, color=c_orig)
        axz.scatter(pos[lo:hi], I_rel[lo:hi], s=25, color=c_rel, alpha=0.6)
        # tolerance of motif_pair_acc: the perturbed pointer must lie in I[i] +- delta
        axz.errorbar([i], [I[i]], yerr=delta, color="k", lw=1.5, capsize=10, capthick=1.5, zorder=3)
        axz.annotate(f"$I[i] \\pm \\delta$", (i, I[i] + delta), textcoords="offset points", xytext=(10, -4),
                     ha="left", va="top", fontsize=fs_text)
        axz.scatter([i], [I[i]], s=90, color=c_orig, edgecolors="k", zorder=4)
        axz.scatter([i], [I_rel[i]], s=90, color=c_rel, edgecolors="k", zorder=5)
        diff = abs(I_rel[i] - I[i])
        cmp = "$\\leq$" if diff <= delta else "$>$"
        axz.set_title(f"i={i}: $|\\Delta I|$ = {diff} {cmp} $\\delta$ = {delta}", fontsize=fs_label)
        axz.set_xlim(lo - 0.5, hi - 0.5)
        axz.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        span = max(diff, delta) * 1.25
        axz.set_ylim(I[i] - span, I[i] + span)
        axz.set_xlabel("position", fontsize=fs_label)
        if col == 0:
            axz.set_ylabel("MPI (zoom)", fontsize=fs_label)
        # zoomed area on the MPI panel, kept inside its y range
        y_min, y_max = ax2.get_ylim()
        box_lo, box_hi = max(I[i] - span, y_min), min(I[i] + span, y_max)
        ax2.add_patch(plt.Rectangle((lo, box_lo), hi - lo, box_hi - box_lo, fill=False, ec="k", lw=0.8, ls="--"))
        zooms.append((axz, ((lo + hi) / 2, box_lo)))

    # arrows from each box to its zoom
    for axz, box_bottom in zooms:
        fig.add_artist(ConnectionPatch(xyA=box_bottom, coordsA=ax2.transData,
                                       xyB=(0.5, 1.17), coordsB=axz.transAxes,
                                       arrowstyle="-|>", mutation_scale=15, color="k", lw=1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
