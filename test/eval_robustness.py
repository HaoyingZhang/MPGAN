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

def blockify_mp(mp_array, window_len):
    """
    mp_array: np.ndarray of shape [n_ts, L]
    returns:  np.ndarray of shape [n_ts, L, window_len]
    """
    n_ts, L = mp_array.shape

    pad = window_len - 1
    padded = np.pad(
        mp_array,
        pad_width=((0, 0), (0, pad)),
        mode="constant",
        constant_values=0.0
    )  # [n_ts, L + pad]

    blocks = np.zeros((n_ts, L, window_len), dtype=np.float32)

    for i in range(L):
        blocks[:, i, :] = padded[:, i:i + window_len]

    return blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='===== GAN to generate synthetic time series with the given Matrix Profile =====')
    parser.add_argument("-n_ts", type=int, required=True, help="Length of the dataset")
    parser.add_argument("-n", type=int, required=True, help="Length of the considered time series")
    parser.add_argument("-m", type=int, required=True, help="Subsequence length for the Matrix Profile")
    parser.add_argument("-e", type=int, default = 10, help="Number of epoches to train the network")
    parser.add_argument("-r", "--random_seed", type=int, default=None, help="Random seed used to generate the time series (default: None)")
    parser.add_argument("-c", "--category", type=str, default="theoretical", help="Category of the time series: 'theoretical', 'energy', or 'ecg' (default: 'theoretical')")
    parser.add_argument("-train_id", "--train_id", type=int, nargs="+", help="IDs of persons used in the training set")
    parser.add_argument("-test_id", "--test_id", type=int, nargs="+", help="IDs of persons used in the test set")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the original time series and the solutions (default: False)")
    parser.add_argument("-k", type=float, default = 1.0, help="Focus on optimizing the top k percent small distance, default : the original objective function")
    parser.add_argument("-g_model", type=str, default = "lstm", help="Choose the G model")
    parser.add_argument("-obj_func", type=str, default = "relu", help="Define the objective function used in the training")
    parser.add_argument("-alpha", type=float, default = 0.05, help="Define the parameter used in exponential objective function")
    parser.add_argument("-pi_mp", type=float, default = 0.05, help="Define the coefficient of the condition loss")
    parser.add_argument("-pi_mse", type=float, default = 0.05, help="Define the coefficient of the original MSE loss")
    parser.add_argument("-pi_pcc", type=float, default = 0.05, help="Define the coefficient of the original PCC loss")
    parser.add_argument("-pi_grad", type=float, default = 0.05, help="Define the coefficient of the original Temporal Gradiant loss")
    parser.add_argument("-latent", "--enable_latent", action="store_true", help="Latent dimension")
    parser.add_argument("-mp_norm", "--enable_mp_norm", action="store_true", help="Enable normalized MP")
    parser.add_argument("-lr_g", type=float, default=1e-5, help="Learning rate for Generator")
    parser.add_argument("-coeff_dist", type=float, default = 1.0, help="Define the coefficient of the distance loss in MP")
    parser.add_argument("-coeff_index", type=float, default = 1.0, help="Define the coefficient of the index loss in MP")
    parser.add_argument("-time", type=int, default = None, help="Time limit" )
    parser.add_argument("-inj_proj", "--enable_inj_proj", action="store_true", help="Using projection to expand features")
    parser.add_argument("-do", "--enable_drop_out", action="store_true", help="Enable the drop out layer")
    parser.add_argument("-mpd_only", "--enable_mpd_only", action="store_true", help="Enable to use MPD only")
    parser.add_argument("-znorm", "--znorm_mp", action="store_true", help="Using z-normalized Euclidean distance in MP computing")
    parser.add_argument("-test", "--test_set_enable", action="store_true")
    parser.add_argument("-mp_embedding", "--enable_mp_embedding", action="store_true", help="Using matrix embedding for the MP input")
    parser.add_argument("-fill", "--fill_value", type=float, default = 100.0, help="The value to fill in the MP embedding")
    parser.add_argument("-dataset", type=str, default = "ltdb", help="The ECG dataset used")
    parser.add_argument("-rob", type=str, default="mpi", help="The perturbation list")
    parser.add_argument("-eps", type=float, default=100.0, help="The global epsilon (privacy budget)")

    args = parser.parse_args()

    if args.obj_func not in ["relu", "exp"]: 
        parser.error(f"Must choose objective function between relu or exp, but got {args.obj_func}")
    if args.enable_mp_embedding and args.enable_mp_norm:
        parser.error(f"The MP embedding and MP norm cannot be enabled in the same time")

    m = args.m
    n = args.n
    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
    if args.dataset == "ltdb":
        print("using dataset ltdb")
        data_train_dir = "data/physionet.org/files/ltdb/records100/"
        data_test_dir = data_train_dir
        with open(os.path.join(data_train_dir, "RECORDS"), "r") as f:
            id_patient = f.read().split("\n")
        list_patient = [os.path.join(data_train_dir, id+".npy") for id in id_patient]
        files = list_patient[args.train_id[0]: args.train_id[1]]
        files_test = list_patient[args.test_id[0]: args.test_id[1]]

    elif args.dataset == "ptbxl":
        print("Using ptbxl dataset")
        data_train_dir = "data/physionet.org/files/"
        data_test_dir = "data/physionet.org/files/"
        list_patient = pd.read_csv(os.path.join(data_train_dir, "ptbxl_database.csv"))["filename_lr"]
        files = list_patient[args.train_id[0]:args.train_id[1]]
        files = [os.path.join(data_train_dir, file) for file in files]
        files_test = list_patient[args.test_id[0]:args.test_id[1]]
        files_test = [os.path.join(data_test_dir, file) for file in files_test]

    elif args.dataset == "arrhythmia":
        print("Using arrhythmia dataset")
        data_train_dir = "data/physionet.org/files/ecg-arrhythmia/records100/"
        data_test_dir = data_train_dir
        with open(os.path.join(data_train_dir, "RECORDS"), "r") as f:
            id_patient = f.read().split("\n")
        list_patient = [os.path.join(data_train_dir, id+".npy") for id in id_patient]
        files = list_patient[args.train_id[0]: args.train_id[1]]
        files_test = list_patient[args.test_id[0]: args.test_id[1]]
    
    elif args.dataset == "arrhythmia":
        print("Using arrhythmia dataset")
        data_train_dir = "data/physionet.org/files/ecg-arrhythmia/records100/"
        data_test_dir = data_train_dir
        with open(os.path.join(data_train_dir, "RECORDS"), "r") as f:
            id_patient = f.read().split("\n")
        list_patient = [os.path.join(data_train_dir, id+".npy") for id in id_patient]
        files = list_patient[args.train_id[0]: args.train_id[1]]
        files_test = list_patient[args.test_id[0]: args.test_id[1]]

    elif args.dataset == "t-drive":
        print("Using t-drive dataset")
        data_train_dir = "data/T-drive/release/grid/"
        data_test_dir = data_train_dir
        id_patient = np.arange(9105)
        list_patient = [os.path.join(data_train_dir, str(id)+".npy") for id in id_patient]
        files = list_patient[args.train_id[0]: args.train_id[1]]
        files_test = list_patient[args.test_id[0]: args.test_id[1]]

    n_person_training = len(files)
    n_ts_per_person_train = args.n_ts // n_person_training
    n_person_test = len(files_test)
    n_ts_per_person_test = args.n_ts // n_person_test


    L = args.n - m + 1
    window_len = 100
    if args.enable_mpd_only:
        C = window_len
    else:
        C = 2
    
    hidden_dim = 64
    if args.enable_mp_embedding:
        mp_dim = L
    else:
        mp_dim = C  # MPD + MPI
    if args.g_model == "WillBeNamed":
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
    
    # model_save_path = f"src/results/baseline/2026-03-04_10:10:37/"
    # model_save_path = f"src/results/baseline/2026-04-20_13:56:19/"
    model_save_path = f"src/results/baseline/ptbxl/"
    
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
    result_save_path = os.path.join(model_save_path, args.dataset, args.rob, str(args.eps)) if args.test_set_enable else f"test/results/{time_str}/"
    os.makedirs(result_save_path, exist_ok=True)
    save_args(args, output_dir=result_save_path)
    
    G.load_state_dict(
        torch.load(model_save_path + "best_model.pth", map_location="cpu")
    )
    G = G.cpu()

    # 4. Generate samples
    G.eval()
    
    batch_size = 64

    max_index_list = [10828800, 6420480, 10997760, 9454080, 9753600, 10252800, 10237440]
    if args.dataset == "ltdb":
        if args.test_set_enable:
            max_index = np.min([max_index_list[ind] for ind in range(args.test_id[0], args.test_id[1])])
            # max_index = 520000
        else:
            max_index = np.min([max_index_list[ind] for ind in range(args.train_id[0], args.train_id[1])])
        max_index = max_index*100/128
    elif args.dataset == "ptbxl":
        max_index = 1000
    elif args.dataset == "arrhythmia":
        max_index = 180556
    elif args.dataset == "t-drive":
        max_index = 500

    np.random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)
    max_start = max_index - n
    assert max_start >= 0, "Signal shorter than window length"

    n_samples = n_ts_per_person_test if args.test_set_enable else n_ts_per_person_train
    max_possible = max_index // n + 1

    # candidates = np.arange(0, max_start, n, dtype=np.int64)
    # indices_ts = rng.choice(candidates, size=n_samples, replace=False)
    # indices_ts = sample_far_indices(max_start-1, indices_ts[:n_ts_per_person_train],n, 200, rng)

    if n_samples > max_possible:
        print("Not enough room for spaced sampling, overlapped time series will be used")
        indices_ts = np.random.randint(0, max_start, size=n_samples)
    else:
        print("No overlapped time series used")
        candidates = np.arange(0, max_start+1, n, dtype=np.int64)
        indices_ts = rng.choice(candidates, size=n_samples, replace=False)

    # if args.test_set_enable:
    #     indices_ts = indices_ts[n_ts_per_person_train:]
    #     np.save("test_indices.npy", indices_ts)
    # else:
    #     indices_ts = indices_ts[:n_ts_per_person_train]
        # np.save("train_indices.npy", indices_ts)
    if args.dataset == "arrhythmia":
        indices_ts = [0,500,1000,2000,2500]
    elif args.dataset in ("ptbxl", "t-drive"):
        indices_ts = [500]
    
    print(f"Indices : {indices_ts}")

    X_test_full, y_test_full = [], []
    if args.test_set_enable:
        files = files_test
    for file in files:
        if args.dataset in ("arrhythmia", "ltdb", "t-drive"):
            signal = np.load(file)
        else:
            record = wfdb.rdrecord(file)
            signal = record.p_signal[:, 0]
            
        if len(signal) < np.max(indices_ts)+args.n:
            print("Time series not long enough")
            continue

        for start_idx in indices_ts:
            ts = signal[start_idx : start_idx + args.n]
            if len(ts) < args.n:
                print(f"The trancation at index {start_idx} is ignored due to the limitation size in time series")
                continue

            ts = normalize(ts)
            y_test_full.append(ts)
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
                epsilon=args.eps
            )
            for i in range(len(y_test_full))
        ]
    _lines, distance_utility_list = zip(*_results)
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
    plot_res(result_save_path, test_labels, fake_data, distance_utility_list, test_file_names, args.m, args.znorm_mp, plot=args.plot)
    
    # 5. Evaluate Utility: Matrix Profile
    # utility_score = np.mean([compute_matrix_profile_distance(real_x.squeeze(), fake_x.squeeze()) for real_x, fake_x in zip(test_set, fake_data)])
    # print(f"✅ Matrix Profile Distance (lower is better): {utility_score:.4f}")

    # 6. TODO : Evaluate Privacy: Membership Inference Attack
    # mia_score = run_mia_attack(member_test, non_member_test, fake_data)
    # print(f"🔒 MIA Attack Accuracy (higher = less private): {mia_score:.4f}")

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