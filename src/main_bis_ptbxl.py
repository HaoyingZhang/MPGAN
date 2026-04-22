# GLOBAL IMPORTS
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os, argparse, glob, json
os.environ["NUMBA_DISABLE_CUDA"] = "1"
import stumpy
from scipy import stats
from dataloader import MemmapDataset
import wfdb
from concurrent.futures import ProcessPoolExecutor
import time
import pandas as pd
import mne

# LOCAL IMPORTS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # Add root directory to path
from models.LSTM import Generator as G_LSTM
from models.BiLSTM import Generator as G_biLSTM
# from models.LSTM import Discriminator
# from models.BiLSTM import Pulse2pulseDiscriminator
from models.Transformer import MPEncoderDecoder as G_Transformer
from models.CNN import Discriminator as D
from models.CNN import Generator as G_CNN
from models.WillBeNamed import Generator as G_WillBeNamed
from training.train_baseline import train_gan, train_wgan_gp, train_inverse
from src.utils_matrix_profile import compute_matrix_profile_distance, MP_compute_recursive, MP_compute_single, build_mp_embedding_batch
from training.objectives import objective_function_pytorch, objective_function_exponential_pytorch, objective_function_unified

def normalize(time_series : np.ndarray) -> np.ndarray:
    rng = time_series.max() - time_series.min()
    if rng == 0:
        return np.zeros_like(time_series, dtype=np.float64)
    return (time_series - time_series.min()) / rng

def pearson_correlation(x, y):
    """Compute the absolute Pearson correlation between two time series x and y."""
    if np.all(x == y) or np.all(x == -y):
        return 1.0
    elif np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    val = stats.pearsonr(x, y).statistic 
    return val if val>=0 else -val

def plot_res(folder_name, real_data, fake_data, name_list, m, mp_norm):
    original_color = 'tab:blue'
    fake_color = 'tab:red'
    for real_ts, fake_ts, file_name in zip(real_data, fake_data, name_list):
        file_name_short = os.path.splitext(os.path.basename(file_name))[0]
        real_ts = real_ts.numpy().squeeze().astype(np.float64)
        fake_ts = fake_ts.numpy().squeeze().astype(np.float64)
        fake_ts = normalize(fake_ts)
        os.makedirs(os.path.join(folder_name, file_name_short), exist_ok=True)
        res_json = {
            "data": real_ts.tolist(),
            "fake_data": fake_ts.tolist()
        }

        with open(os.path.join(folder_name, file_name_short, "results.json"), "w") as f:
            json.dump(res_json, f, indent=4)

        mp_real = stumpy.stump(real_ts, m, normalize=mp_norm, ignore_trivial=True)
        mp_fake = stumpy.stump(fake_ts, m, normalize=mp_norm, ignore_trivial=True)
        mp_real_clean = mp_real[:, [0, 1]].astype(np.float32)
        mp_real_clean = np.nan_to_num(mp_real_clean, nan=0.0, posinf=0.0, neginf=0.0)
        mp_fake_clean = mp_fake[:, [0, 1]].astype(np.float32)
        mp_fake_clean = np.nan_to_num(mp_fake_clean, nan=0.0, posinf=0.0, neginf=0.0)
        figsize=(19.2,9)
        fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, squeeze=False)

        # Plot the time series
        axs[0, 0].plot(real_ts, label='Original TS', color=original_color)
        axs[0, 0].plot(fake_ts, label='Fake TS', color=fake_color)
        axs[0, 0].set_title(f"Pearson Corr: {round(pearson_correlation(real_ts, np.array(fake_ts)), 2)}", fontsize=12)
        axs[0, 0].legend(fontsize=10)

        # Plot the Matrix Profile distances
        axs[0, 1].plot(mp_real_clean[:, 0], label='Original MPD', color=original_color)
        axs[0, 1].plot(mp_fake_clean[:, 0], label='Fake MPD', color=fake_color)
        axs[0, 1].set_title(f"Pearson Corr: {round(pearson_correlation(mp_real_clean[:, 0], np.array(mp_fake_clean)[:, 0]), 2)}", fontsize=12)
        axs[0, 1].legend(fontsize=10)

        # Plot the Matrix Profile indices
        axs[1, 1].scatter(np.arange(len(mp_real_clean)), mp_real_clean[:, 1], label='Original MPI', color=original_color)
        axs[1, 1].scatter(np.arange(len(mp_real_clean)), mp_fake_clean[:, 1], label='Fake MPI', color=fake_color)
        axs[1, 1].set_title(f"Accuracy: {round(np.sum([x==y for x, y in zip(mp_real_clean[:, 1], mp_fake_clean[:, 1])]))}/{len(mp_real)}", fontsize=12)
        axs[1, 1].legend(fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(folder_name, file_name_short, "results.png"))
        plt.close(fig)

def save_args(args, output_dir="src/baseline/results/", filename="config.json"):
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    config_path = os.path.join(output_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Saved configuration to {config_path}")

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
    parser = argparse.ArgumentParser(description='===== Inverse with the given Matrix Profile =====')
    parser.add_argument("-n_ts", type=int, default=None, help="Max dataset size per person (optional; all valid slices used when omitted)")
    parser.add_argument("-n", type=int, required=True, help="Length of the considered time series")
    parser.add_argument("-m", type=int, required=True, help="Subsequence length for the Matrix Profile")
    parser.add_argument("-e", type=int, default = 10, help="Number of epoches to train the network")
    parser.add_argument("-r", "--random_seed", type=int, default=None, help="Random seed used to generate the time series (default: None)")
    parser.add_argument("-c", "--category", type=str, default="theoretical", help="Category of the time series: 'theoretical', 'energy', or 'ecg' (default: 'theoretical')")
    parser.add_argument("-dataset", "--dataset", type=str, default="ltdb", help="Dataset name")
    parser.add_argument("-train_id", "--train_id", type=int, nargs="+", help="IDs of persons used in the training set between 0 to 451")
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
    parser.add_argument("-mp_norm", "--enable_mp_norm", action="store_true", help="Enable normalized MP in input")
    parser.add_argument("-lr_g", type=float, default=1e-5, help="Learning rate for Generator")
    parser.add_argument("-coeff_dist", type=float, default = 1.0, help="Define the coefficient of the distance loss in MP")
    parser.add_argument("-coeff_index", type=float, default = 1.0, help="Define the coefficient of the index loss in MP")
    parser.add_argument("-time", type=int, default = None, help="Time limit" )
    parser.add_argument("-inj_proj", "--enable_inj_proj", action="store_true", help="Using projection to expand features")
    parser.add_argument("-do", "--enable_drop_out", action="store_true", help="Enable the drop out layer")
    parser.add_argument("-mpd_only", "--enable_mpd_only", action="store_true", help="Enable to use MPD only")
    parser.add_argument("-znorm", "--znorm_mp", action="store_true", help="Using z-normalized Euclidean distance in MP computing")
    parser.add_argument("-mp_embedding", "--enable_mp_embedding", action="store_true", help="Using matrix embedding for the MP input")
    parser.add_argument("-fill", "--fill_value", type=float, default = 100.0, help="The value to fill in the MP embedding")
    parser.add_argument("-val", "--enable_validation", action="store_true", help="Use validation set in training")
    parser.add_argument("-n_val", type=int, default=1000, help="Number of validation time series (non-overlapping, starting at signal position 20000)")

    args = parser.parse_args()

    if args.obj_func not in ["relu", "exp"]:
        parser.error(f"Must choose objective function between relu or exp, but got {args.obj_func}")
    if args.enable_mp_embedding and args.enable_mp_norm:
        parser.error(f"The MP embedding and MP norm cannot be enabled in the same time")

    m = args.m
    n = args.n

    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    if args.dataset == "ptbxl":
        max_train_index = 1000
    elif args.dataset == "t-drive":
        max_train_index = 500
    elif args.dataset == "tdbrain":
        max_train_index = 60000  # ~5 min at 250 Hz

    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    time_start = datetime.now()
    time_str = time_start.strftime("%Y-%m-%d_%H:%M:%S")

    time_start_dataset = time.time()
    train_epoch = int(args.e)

    if args.category == "ecg":
        data_train_dir = data_test_dir = "data/physionet.org/files/"
    elif args.category == "trajectory":
        data_train_dir = data_test_dir = "data/T-drive/release/grid/"
    elif args.category == "eeg":
        data_train_dir = data_test_dir = "data/TDBRAIN-dataset/"

    if args.dataset == "ptbxl":
        list_patient = pd.read_csv(os.path.join(data_train_dir, "ptbxl_database.csv"))["filename_lr"]
    # elif args.dataset == "arrhythmia":
    #     with open(os.path.join(data_train_dir, "ecg-arrhythmia", "1.0.0", "RECORDS"),"r") as f:
    #         list_patient = [line.strip() for line in f.readlines()]
    # elif args.dataset == "ltdb":
    elif args.dataset == "t-drive":
        list_patient = [os.path.join(data_train_dir,str(i)+".npy") for i in np.arange(9105)]
    elif args.dataset == "tdbrain":
        list_patient = pd.read_csv(os.path.join(data_train_dir, "participants.tsv"), sep="\t")["participant_id"].tolist()

    files = list_patient[args.train_id[0]:args.train_id[1]]
    files_test = list_patient[args.test_id[0]:args.test_id[1]]
    print(files)
    n_person_training = len(files)

    L = args.n - m + 1
    window_len = 100
    if args.enable_mpd_only:
        C = window_len
    else:
        C = 2

    X_path = "X_train_n"+str(n)+"_m"+str(m)+"_p"+str(args.train_id)+".dat"
    y_path = "y_train_n"+str(n)+"_m"+str(m)+"_p"+str(args.train_id)+".dat"

    float32_bytes = np.dtype(np.float32).itemsize
    X_row_bytes = L * C * float32_bytes
    y_row_bytes = n * float32_bytes

    # Try to reuse cached memmaps: infer actual_n_ts from file sizes
    X_cached = os.path.exists(X_path)
    y_cached = os.path.exists(y_path)
    actual_n_ts = None
    if X_cached and y_cached:
        X_rows = os.path.getsize(X_path) // X_row_bytes
        y_rows = os.path.getsize(y_path) // y_row_bytes
        if X_rows == y_rows and X_rows > 0:
            actual_n_ts = X_rows
            print(f"[✓] Reusing existing memmaps ({actual_n_ts} samples)")
            X_train_mm = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(actual_n_ts, L, C))
            y_train_mm = np.memmap(y_path, dtype=np.float32, mode="r+", shape=(actual_n_ts, n))
        else:
            print(f"[!] Cached memmaps have inconsistent sizes, recreating.")
            os.remove(X_path); os.remove(y_path)
            X_cached = y_cached = False

    if actual_n_ts is None:
        np.random.seed(args.random_seed)
        rng = np.random.default_rng(args.random_seed)
        max_start = max_train_index - n + 1
        assert max_start > 0, "Signal shorter than window length"
        candidates = np.arange(0, max_start, n, dtype=np.int64)

        if args.n_ts is not None:
            n_ts_per_person = args.n_ts // n_person_training
            if n_ts_per_person == 0:
                raise ValueError("n_ts too small for the number of persons")
            if n_ts_per_person > len(candidates):
                print("Not enough room for spaced sampling, overlapped time series will be used")
                indices_ts = np.random.randint(0, max_start, size=n_ts_per_person)
            else:
                print("No overlapped time series used")
                indices_ts = rng.choice(candidates, size=n_ts_per_person, replace=False)
        else:
            print("No n_ts specified, using all non-overlapping slices")
            indices_ts = candidates

        y_train = []
        for file in files:
            if args.dataset == "t-drive":
                signal = np.load(file)
            elif args.dataset == "tdbrain":
                vhdr_path = os.path.join(data_train_dir, file, "ses-1", "eeg", f"{file}_ses-1_task-restEC_eeg.vhdr")
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
                signal = raw.get_data(picks=0)[0].astype(np.float32)
            else:
                record = wfdb.rdrecord(os.path.join(data_train_dir, file))
                signal = record.p_signal[:, 0].astype(np.float32, copy=False)

            for start_idx in indices_ts:
                ts = signal[start_idx : start_idx + n]
                if len(ts) < n:
                    print(f"Skipping short slice (len={len(ts)}<{n}) in {file} at {start_idx}")
                    continue
                y_train.append(normalize(ts).astype(np.float32, copy=False))

        print(f"Collected {len(y_train)} valid time series")
        X_train = np.stack([
            MP_compute_single(
                y_train[i], m,
                norm=args.enable_mp_norm,
                mpd_only=args.enable_mpd_only,
                znorm=args.znorm_mp,
                fill_value=args.fill_value
            )
            for i in range(len(y_train))
        ])

        actual_n_ts = len(y_train)
        print(f"[+] Creating memmaps for {actual_n_ts} samples")
        X_train_mm = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(actual_n_ts, L, C))
        y_train_mm = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(actual_n_ts, n))
        X_train_mm[:] = X_train
        y_train_mm[:] = np.array(y_train, dtype=np.float32)
        X_train_mm.flush()
        y_train_mm.flush()

    loading_time = time.time() - time_start_dataset
    print(f"Time loading the training data: {loading_time} seconds")
    batch_size = 64

    full_train_dataset = MemmapDataset(
        X_path,
        y_path,
        shape_X=(actual_n_ts, L, C),
        shape_y=(actual_n_ts, args.n)
    )
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Training set length: {len(full_train_dataset)}")

    # ===== Validation set =====
    if args.enable_validation:
        np.random.seed(args.random_seed)
        rng = np.random.default_rng(args.random_seed)
        n_person_val = len(files_test)
        val_candidates = np.arange(0, max_train_index - n + 1, n, dtype=np.int64)

        if args.n_val is not None:
            n_val_per_person = args.n_val // n_person_val
            if n_val_per_person == 0:
                raise ValueError(f"n_val={args.n_val} too small for {n_person_val} test persons")
            if n_val_per_person > len(val_candidates):
                raise ValueError(f"Not enough non-overlapping positions in [0, {max_train_index}) for {n_val_per_person} samples/person")
            indices_val = rng.choice(val_candidates, size=n_val_per_person, replace=False)
        else:
            indices_val = val_candidates

        X_val_path = "X_val_n"+str(n)+"_m"+str(m)+"_p"+str(args.test_id)+".dat"
        y_val_path = "y_val_n"+str(n)+"_m"+str(m)+"_p"+str(args.test_id)+".dat"

        X_val_cached = os.path.exists(X_val_path)
        y_val_cached = os.path.exists(y_val_path)
        n_val_total = None
        if X_val_cached and y_val_cached:
            X_val_rows = os.path.getsize(X_val_path) // X_row_bytes
            y_val_rows = os.path.getsize(y_val_path) // y_row_bytes
            if X_val_rows == y_val_rows and X_val_rows > 0:
                n_val_total = X_val_rows
                print(f"[✓] Reusing existing val memmaps ({n_val_total} samples)")
                X_val_mm = np.memmap(X_val_path, dtype=np.float32, mode="r+", shape=(n_val_total, L, C))
                y_val_mm = np.memmap(y_val_path, dtype=np.float32, mode="r+", shape=(n_val_total, n))
            else:
                print(f"[!] Cached val memmaps have inconsistent sizes, recreating.")
                os.remove(X_val_path); os.remove(y_val_path)
                X_val_cached = y_val_cached = False

        if n_val_total is None:
            y_val_list = []
            for file in files_test:
                if args.dataset == "t-drive":
                    signal = np.load(file)
                elif args.dataset == "tdbrain":
                    matches = glob.glob(os.path.join(data_test_dir, file, "ses-1", "eeg", "*restEC*.vhdr"))
                    raw = mne.io.read_raw_brainvision(matches[0], preload=True, verbose=False)
                    signal = raw.get_data(picks=0)[0].astype(np.float32)
                else:
                    record = wfdb.rdrecord(os.path.join(data_test_dir, file))
                    signal = record.p_signal[:, 0].astype(np.float32, copy=False)
                for start_idx in indices_val:
                    ts = signal[start_idx : start_idx + n]
                    if len(ts) < n:
                        print(f"Skipping short val slice (len={len(ts)}<{n}) in {file} at {start_idx}")
                        continue
                    y_val_list.append(normalize(ts).astype(np.float32, copy=False))

            n_val_total = len(y_val_list)
            print(f"Collected {n_val_total} valid validation time series")
            X_val_arr = np.stack([
                MP_compute_single(
                    y_val_list[i], m,
                    norm=args.enable_mp_norm,
                    mpd_only=args.enable_mpd_only,
                    znorm=args.znorm_mp,
                    fill_value=args.fill_value
                )
                for i in range(n_val_total)
            ])
            X_val_mm = np.memmap(X_val_path, dtype=np.float32, mode="w+", shape=(n_val_total, L, C))
            y_val_mm = np.memmap(y_val_path, dtype=np.float32, mode="w+", shape=(n_val_total, n))
            X_val_mm[:] = X_val_arr
            y_val_mm[:] = np.array(y_val_list, dtype=np.float32)
            X_val_mm.flush()
            y_val_mm.flush()

        val_dataset = MemmapDataset(
            X_val_path, y_val_path,
            shape_X=(n_val_total, L, C),
            shape_y=(n_val_total, args.n)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"Validation set length: {len(val_dataset)}")
    else:
        val_loader = None

    # Initialize model hyperparameters
    batch_size, n_input, window_len = next(iter(train_loader))[0].shape  # n_input = 2*(n-m+1)
    
    hidden_dim = 64
    if args.enable_mp_embedding:
        mp_dim = L
    else:
        mp_dim = C  # MPD + MPI

    # assert n_input == 2*(n-m+1)

    # Enable latent
    if args.enable_latent:
        latent_dim = 16
    else:
        latent_dim = None
    # Models
    if args.g_model == "lstm":
        G = G_LSTM(input_dim=mp_dim, hidden_dim=hidden_dim, output_length=n)
    elif args.g_model == "bi-lstm":
        G = G_biLSTM(n=n, m=m)
    elif args.g_model == "cnn":
        G = G_CNN(n=n, m=m)
    elif args.g_model == "WillBeNamed":
        G = G_WillBeNamed(
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
    elif args.g_model == "transformer":
        G = G_Transformer(n=n, m=m, d_model=args.m, nhead=5)

    # 3. Train GAN
    model_save_path = f"src/results/baseline/{time_str}/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    save_args(args, model_save_path, "config.json")
    G, G_loss, best_val_loss = train_inverse(train_loader,
                                                val_loader, 
                                                 G, 
                                                 device=device, 
                                                 checkpoint_path=model_save_path, 
                                                 epoch=train_epoch, 
                                                 mp_window_size=args.m, 
                                                 k_violation=args.k, 
                                                 alpha=args.alpha, 
                                                 activ_func=args.obj_func, 
                                                 time_limit=args.time,
                                                 pi_mp=args.pi_mp,
                                                 pi_mse=args.pi_mse,
                                                 pi_pcc=args.pi_pcc,
                                                 pi_grad=args.pi_grad,
                                                 lr_G=args.lr_g,
                                                 latent=args.enable_latent,
                                                 coeff_dist = args.coeff_dist,
                                                 coeff_identity=args.coeff_index,
                                                 mp_norm=args.znorm_mp,
                                                 embedding_mp=args.enable_mp_embedding,
                                                 fill_value=args.fill_value)

    G.load_state_dict(torch.load(model_save_path+"best_model.pth"))
    G = G.cpu()
    # 4. Generate samples
    G.eval()

    if args.enable_validation:
        plot_tensor = torch.tensor(X_val_mm[:], dtype=torch.float32)
        if args.enable_mp_embedding:
            plot_tensor = build_mp_embedding_batch(
                plot_tensor[..., 0],
                plot_tensor[..., 1],
                fill_value=args.fill_value
            )
        plot_labels = torch.tensor(y_val_mm[:], dtype=torch.float32)
        with torch.no_grad():
            if args.enable_latent:
                z = torch.randn(plot_tensor.size(0), 64, device='cpu') if G.z_dim else None
                fake_data = G(plot_tensor, z=z)
            else:
                fake_data = G(plot_tensor)
        plot_file_names = [f"{args.category}_"+str(i) for i in range(len(plot_tensor))]
        if args.plot:
            plot_res(model_save_path, plot_labels, fake_data, plot_file_names, args.m, args.znorm_mp)
    
    # 5. Evaluate Utility: Matrix Profile
    # utility_score = np.mean([compute_matrix_profile_distance(real_x.squeeze(), fake_x.squeeze()) for real_x, fake_x in zip(test_set, fake_data)])
    # print(f"✅ Matrix Profile Distance (lower is better): {utility_score:.4f}")

    # 6. TODO : Evaluate Privacy: Membership Inference Attack
    # mia_score = run_mia_attack(member_test, non_member_test, fake_data)
    # print(f"🔒 MIA Attack Accuracy (higher = less private): {mia_score:.4f}")

    # Plot the first feature (D=1 assumed) or slice index 0
    if args.enable_validation:
        plt.figure(figsize=(5, 12))
        for i in range(5):
            plt.subplot(5, 1, i + 1)
            plt.plot(fake_data[i, :].detach().numpy(), label=f"Fake Sample {i}")
            plt.plot(normalize(plot_labels[i].numpy()), label=f"Real Sample {i}")
            plt.legend()
            plt.tight_layout()

        plt.suptitle("Generated Time Series (First Feature)", fontsize=16, y=1.02)
        plt.savefig(os.path.join(model_save_path, "time_series.png"))
        plt.close()

    # 7. Plot the loss curves
    # d_loss_list_np = np.array(d_loss_list)
    g_loss_list_np = np.array(G_loss['train_G'])
    plt.figure(figsize=(8, 5))
    # plt.plot(d_loss_list, label="Discriminator Loss", marker='o')
    plt.plot(g_loss_list_np, label="Generator Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, "loss.png"))
    plt.close()
    