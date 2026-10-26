# GLOBAL IMPORTS
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os, argparse, glob, json
import stumpy
from scipy import stats
import wfdb
import pandas as pd

# LOCAL IMPORTS
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)  # Add root directory to path

from src.utils_matrix_profile import MP_compute_single
from src.models.WillBeNamed import Generator
from src.models.Transformer import MPEncoderDecoder as ED_Generator
from src.utils_matrix_profile import MP_compute_recursive
from main import normalize, plot_res, save_args
from src.utils import pearson_correlation, mp_utility_metrics_motif, mp_utility_metrics_discord
from reidentify import reidentification_attack
from loader import ptbxl_loader, arrhythmia_loader, ltdb_loader, tdbrain_loader, ptb_diagnostic_loader, ptb_diagnostic_records_per_patient, eegmmidb_loader, eegmmidb_subjects

DATA_LOADERS = {"ptbxl": ptbxl_loader, "arrhythmia": arrhythmia_loader, "arrhythmia_xl": arrhythmia_loader, "ltdb": ltdb_loader, "tdbrain": tdbrain_loader, "ptb_diagnostic": ptb_diagnostic_loader, "eegmmidb": eegmmidb_loader}
LIST_PEOPLE =  {"ptbxl": np.arange(21000,21200), "arrhythmia": np.arange(48), "arrhythmia_xl": np.arange(48), "ltdb": np.arange(7), "tdbrain": np.arange(1000,1200), "ptb_diagnostic": np.arange(len(ptb_diagnostic_records_per_patient())), "eegmmidb": np.arange(len(eegmmidb_subjects()))}
# ptb_diagnostic is recorded at 1000 Hz but, as in test/eval.py, resampled to 100 Hz
SOURCE_HZ = {"ptbxl": 100, "arrhythmia": 100, "arrhythmia_xl": 100, "ltdb": 100, "tdbrain": 500, "ptb_diagnostic": 100, "eegmmidb": 160}
SESSION_EEG = "EC"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='===== GAN to generate synthetic time series with the given Matrix Profile =====')
    parser.add_argument("-n_ts", type=int, required=True, help="Number of people evaluated (the first n_ts of the dataset)")
    parser.add_argument("-n", type=int, required=True, help="Length of the considered time series")
    parser.add_argument("-m", type=int, required=True, help="Subsequence length for the Matrix Profile")
    parser.add_argument("-r", "--random_seed", type=int, default=None, help="Random seed used to generate the time series (default: None)")
    parser.add_argument("-c", "--category", type=str, default="theoretical", help="Category of the time series: 'theoretical', 'energy', or 'ecg' (default: 'theoretical')")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the original time series and the solutions (default: False)")
    parser.add_argument("-g_model", type=str, default = "lstm", help="Choose the G model")
    parser.add_argument("-latent", "--enable_latent", action="store_true", help="Latent dimension")
    parser.add_argument("-mp_norm", "--enable_mp_norm", action="store_true", help="Enable normalized MP")
    parser.add_argument("-inj_proj", "--enable_inj_proj", action="store_true", help="Using projection to expand features")
    parser.add_argument("-do", "--enable_drop_out", action="store_true", help="Enable the drop out layer")
    parser.add_argument("-mpd_only", "--enable_mpd_only", action="store_true", help="Enable to use MPD only")
    parser.add_argument("-znorm", "--znorm_mp", action="store_true", help="Using z-normalized Euclidean distance in MP computing")
    parser.add_argument("-mp_embedding", "--enable_mp_embedding", action="store_true", help="Using matrix embedding for the MP input")
    parser.add_argument("-fill", "--fill_value", type=float, default = 100.0, help="The value to fill in the MP embedding")
    parser.add_argument("-dataset", type=str, default = "ltdb", help="The ECG dataset used")
    parser.add_argument("-resample_hz", type=int, default=None, metavar="HZ",help="Resample signals to this rate (Hz) before processing")
    parser.add_argument("--rob", type=str, default="mpi", help="The perturbation list")
    parser.add_argument("--eps", type=float, default=100.0, help="The global epsilon (privacy budget)")
    parser.add_argument("--eps_mpi", type=float, default=None, help="Epsilon for the MPI perturbation (defaults to --eps)")
    parser.add_argument("--eps_mpd", type=float, default=None, help="Epsilon for the MPD perturbation (defaults to --eps)")
    parser.add_argument("--k", type=int, default=3, help="Number of peaks for the @k utility metrics")

    args = parser.parse_args()

    dataset = args.dataset
    resample_hz = args.resample_hz

    m = args.m
    n = args.n
    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    frequency = SOURCE_HZ[dataset] if resample_hz is None else resample_hz
    people = LIST_PEOPLE[dataset][:args.n_ts]  # first n_ts people
    print(f"Using {len(people)} people of {dataset}")
    if dataset == "tdbrain":
        all_ts = DATA_LOADERS[dataset](people, dest_hz=frequency, session=SESSION_EEG)
    elif dataset == "ptb_diagnostic":
        all_ts = DATA_LOADERS[dataset](people, dest_hz=frequency, one_per_patient=True)
    else:
        all_ts = DATA_LOADERS[dataset](people, dest_hz=frequency)

    L = args.n - m + 1
    window_len = 100
    if args.enable_mpd_only:
        C = window_len
    else:
        C = 2
    
    if args.enable_mp_embedding:
        mp_dim = L
    else:
        mp_dim = C  # MPD + MPI
    if args.g_model == "deepmp":
        G = Generator(
                    n=n,
                    m=m,
                    mp_channels=mp_dim,
                    base_channels=64,
                    num_blocks=6,
                    dilations=(1,2,4,8,16,32),
                    use_attention=True,
                    z_dim=64 if args.enable_latent else None,
                    y_dim=None,         
                    use_in_proj=args.enable_inj_proj,
                    dropout=args.enable_drop_out
                )
    else:
        G = ED_Generator(n=n, m=m, d_model=m, nhead=5)
    
    if args.category == "ecg":
        model_save_path = f"src/results/baseline/ptbxl/"
    else:
        model_save_path = f"src/results/baseline/eeg_160hz/"

    eps_mpi = args.eps if args.eps_mpi is None else args.eps_mpi
    eps_mpd = args.eps if args.eps_mpd is None else args.eps_mpd
    # the folder only names the budget(s) of the perturbed component(s)
    if args.rob == "mpi":
        eps_dir = f"mpi{eps_mpi}"
    elif args.rob == "mpd":
        eps_dir = f"mpd{eps_mpd}"
    elif args.eps_mpi is None and args.eps_mpd is None:
        eps_dir = str(args.eps)
    else:
        eps_dir = f"mpi{eps_mpi}_mpd{eps_mpd}"
    result_save_path = os.path.join(model_save_path, args.dataset, args.rob, eps_dir)
    os.makedirs(result_save_path, exist_ok=True)
    save_args(args, output_dir=result_save_path)
    
    G.load_state_dict(
        torch.load(model_save_path + "best_model.pth", map_location="cpu")
    )
    G = G.cpu()

    # 4. Generate samples
    G.eval()

    if args.dataset == "arrhythmia":
        indices_ts = [0,1000,1500, 2000, 2500]
    elif args.dataset in ("ptbxl", "t-drive", "ptb_diagnostic"):
        indices_ts = [500]
    elif args.dataset in ("tdbrain", "eegmmidb"):
        indices_ts = [500]
    elif args.dataset == "kit":
        indices_ts = [0]
    elif args.dataset == "ltdb":
        indices_ts = [3029500,4160500,1416000,3648500,4532500,
                    132000,4300000,2584000,4846000,3612000, 
                    888000,2338000,4260500,3188000,3774500,
                    830000,1854000,895000,1776500,3270000,
                    1828500,591500,399000,3201000,3220500,
                    3958000,3527000,493000,4611000,1494500]
    
    print(f"Indices : {indices_ts}")

    X_test_full, y_test_full = [], []
    for ts in all_ts:
        for start_idx in indices_ts:
            segment = ts[start_idx : start_idx + args.n]
            if len(segment) < args.n:
                print(f"The truncation at index {start_idx} is ignored due to the limitation size in time series")
                continue
            y_test_full.append(normalize(segment))
    print(f"Loading {len(y_test_full)} time series")

    _results = [
        MP_compute_single(
                y_test_full[i], m,
                norm=args.enable_mp_norm,
                mpd_only=args.enable_mpd_only,
                znorm=args.znorm_mp,
                embedding=args.enable_mp_embedding,
                fill_value=args.fill_value,
                perturb=args.rob,
                epsilon=args.eps,
                epsilon_mpi=args.eps_mpi,
                epsilon_mpd=args.eps_mpd,
                return_mp=True
            )
            for i in range(len(y_test_full))
        ]
    _lines, distance_utility_list, mp_real_list, mp_perturbed_list = zip(*_results)
    X_test_full = np.stack(_lines)

    print("Dataset loaded")
            
    test_tensor = torch.stack([torch.tensor(X_test_full[i], dtype=torch.float32) 
                                for i in range(len(X_test_full))])
    test_labels = torch.stack([torch.tensor(y_test_full[i], dtype=torch.float32) 
                                for i in range(len(y_test_full))])
    with torch.no_grad():
        if args.enable_latent:
            z = torch.randn(test_tensor.size(0), 64, device='cpu') if G.z_dim else None
            fake_data = G(test_tensor, z=z)
        else:
            fake_data = G(test_tensor)
    fake_data = normalize(fake_data)
    test_file_names = [f"{args.category}_"+str(i) for i in range(len(X_test_full))]
    # Plot results
    plot_res(result_save_path, test_labels, fake_data, mp_real_list, mp_perturbed_list, distance_utility_list, test_file_names, args.m, args.znorm_mp, plot=args.plot)

    # Raw results of this run, appended to the previous runs in eval_metrics.json; the means are
    # left to the plotting (src/plot/plot_paper.py). Per-sample metrics are stored as
    # {metric: [[run 1 samples], [run 2 samples], ...]}, the re-identification rate (one value
    # per run) as {"reidentification_rate": [run 1, run 2, ...]}.
    per_sample = {}
    for i in range(len(y_test_full)):
        mp_orig, mp_pert = np.asarray(mp_real_list[i]), np.asarray(mp_perturbed_list[i])
        P, I = mp_orig[:, 0], mp_orig[:, 1].astype(int)
        P_rel, I_rel = mp_pert[:, 0], mp_pert[:, 1].astype(int)
        sample = {"reconstruction_pcc": pearson_correlation(np.asarray(y_test_full[i], dtype=np.float64),
                                                            normalize(fake_data[i].numpy().squeeze().astype(np.float64))),
                  **distance_utility_list[i],
                  **mp_utility_metrics_motif(P, I, P_rel, I_rel, args.m, k=args.k),
                  **mp_utility_metrics_discord(P, P_rel, args.m, k=args.k)}
        for name, v in sample.items():
            per_sample.setdefault(name, []).append(float(v))
    rir = float(reidentification_attack(result_save_path, ipopt=False, dataset=args.dataset, category=args.category,
                                        use_mp=False, classifier="svm", use_feature=True,
                                        use_original=False, distance=False))
    print(f"Re-identification rate: {rir:.4f}")

    metrics_file = os.path.join(result_save_path, "eval_metrics.json")
    metrics_history = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics_history = json.load(f)
    for name, values in per_sample.items():
        metrics_history[name] = metrics_history.get(name, []) + [values]
    metrics_history["reidentification_rate"] = metrics_history.get("reidentification_rate", []) + [rir]
    with open(metrics_file, "w") as f:
        json.dump(metrics_history, f, indent=4)

    # Plot the first feature (D=1 assumed) or slice index 0
    plt.figure(figsize=(5, 12))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(fake_data[i, :], label=f"Fake Sample {i}")
        plt.plot(normalize(test_labels[i]), label=f"Real Sample {i}")
        plt.legend()
        plt.tight_layout()

    plt.suptitle("Generated Time Series (First Feature)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(model_save_path, "time_series.png"))
    plt.close()