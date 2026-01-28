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
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())

def pearson_correlation(x, y):
    """Compute the Pearson correlation between two time series x and y."""
    if np.all(x == y) or np.all(x == -y):
        return 1.0
    elif np.all(x == x[0]) or np.all(y == y[0]):
        return np.NaN
    return stats.pearsonr(x, y).statistic ** 2

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

        mp_real = stumpy.stump(real_ts, m, normalize=mp_norm)
        mp_fake = stumpy.stump(fake_ts, m, normalize=mp_norm)
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

def normalize_ts(ts):
    if ts.size == 0 or ts.shape[0] != args.n:
        return
    min_v = ts.min()
    max_v = ts.max()
    if max_v == min_v:
        return np.zeros_like(ts)
    return (ts - min_v) / (max_v - min_v)

def _process_index_batch(
    signal,
    indices,
    start_row,
    n, m,
    enable_mp_norm,
    enable_mpd_only,
    znorm_mp
):
    X_batch = []
    y_batch = []

    for start_idx in indices:
        ts = signal[start_idx : start_idx + n]
        ts = normalize(ts).astype(np.float32, copy=False)

        mp = MP_compute_single(
            ts, m,
            norm=enable_mp_norm,
            mpd_only=enable_mpd_only,
            znorm=znorm_mp
        )
        y_batch.append(ts)
        X_batch.append(mp)

    return start_row, np.stack(X_batch), np.stack(y_batch)

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

    args = parser.parse_args()

    if args.obj_func not in ["relu", "exp"]: 
        parser.error(f"Must choose objective function between relu or exp, but got {args.obj_func}")
    if args.enable_mp_embedding and args.enable_mp_norm:
        parser.error(f"The MP embedding and MP norm cannot be enabled in the same time")

    m = args.m
    n = args.n
    max_index = 10828800 # TODO: define the max_start from all the ts 
    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    time_start = datetime.now()
    time_str = time_start.strftime("%Y-%m-%d_%H:%M:%S")
    
    time_start_dataset = time.time()
    train_epoch = int(args.e)

    list_patient = [14046, 14134, 14149, 14157, 14172, 14184, 15814]

    data_dir = "data/physionet.org/files/ltdb/1.0.0/"
    files = sorted([os.path.join(data_dir, str(list_patient[i])) for i in args.train_id])
    files_test = sorted([os.path.join(data_dir, str(list_patient[i])) for i in args.test_id])

    n_person_training = len(args.train_id)
    n_person_test = len(args.test_id)
    n_ts_per_person_train = args.n_ts // n_person_training
    n_ts_per_person_test = 70

    L = args.n - m + 1
    window_len = 100
    if args.enable_mpd_only:
        C = window_len
    else:
        C = 2

    # Preallocate OUTPUT memmaps
    X_path = "X_train_n"+str(n)+"_m"+str(m)+"_p"+str(args.train_id)+".dat"
    y_path = "y_train_n"+str(n)+"_m"+str(m)+"_p"+str(args.train_id)+".dat"

    X_shape = (args.n_ts, L, C)
    y_shape = (args.n_ts, args.n)

    X_expected_bytes = np.prod(X_shape) * np.dtype(np.float32).itemsize
    y_expected_bytes = np.prod(y_shape) * np.dtype(np.float32).itemsize


    def memmap_or_create(path, shape, dtype):
        file_exist = False
        expected_bytes = np.prod(shape) * np.dtype(dtype).itemsize

        if os.path.exists(path):
            actual_bytes = os.path.getsize(path)

            if actual_bytes == expected_bytes:
                print(f"[✓] Reusing existing memmap: {path}")
                file_exist = True
                return np.memmap(path, dtype=dtype, mode="r+", shape=shape), file_exist
            else:
                print(
                    f"[!] Size mismatch for {path}: "
                    f"expected {expected_bytes}, got {actual_bytes}. Recreating."
                )
                os.remove(path)

        print(f"[+] Creating memmap: {path}")
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape),file_exist


    X_train_mm, X_train_exist = memmap_or_create(X_path, X_shape, np.float32)
    y_train_mm, y_train_exist = memmap_or_create(y_path, y_shape, np.float32)

    print("allocation finished")

    np.random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)
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
    indices_ts_test = indices_ts[n_ts_per_person_train:]
    
    # print(files)
    # print(files_test)
    # print(indices_ts_train)
    # print(indices_ts_test)

    if (not X_train_exist) and (not y_train_exist):

        n_workers = min(os.cpu_count(), len(indices_ts_train))
        index_batches = np.array_split(indices_ts_train, n_workers)
        
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            write_idx = 0

            for file in files:
                record = wfdb.rdrecord(file)
                signal = record.p_signal[:, 0].astype(np.float32, copy=False)
                print(len(signal))

                futures = []
                offset = 0

                for batch in index_batches:
                    futures.append(
                        ex.submit(
                            _process_index_batch,
                            signal=signal,
                            indices=batch,
                            start_row=write_idx + offset,
                            n=args.n,
                            m=m,
                            enable_mp_norm=args.enable_mp_norm,
                            enable_mpd_only=args.enable_mpd_only,
                            znorm_mp=args.znorm_mp
                        )
                    )
                    offset += len(batch)

                for f in futures:
                    start_row, X_batch, y_batch = f.result()
                    X_train_mm[start_row:start_row+len(X_batch)] = X_batch
                    y_train_mm[start_row:start_row+len(y_batch)] = y_batch
                X_train_mm.flush()
                y_train_mm.flush()
                
                write_idx += len(indices_ts_train)

    loading_time = time.time() - time_start_dataset
    print(f"Time loading the training data: {loading_time} seconds")
    batch_size = 64
    
    full_train_dataset = MemmapDataset(
        X_path,
        y_path,
        shape_X=(args.n_ts, L, C),
        shape_y=(args.n_ts, args.n)
    )
    if args.enable_validation:
        # Split sizes
        n_total = len(full_train_dataset)
        n_val = int(0.3 * n_total)
        n_train = n_total - n_val

        # Deterministic split (recommended)
        generator = torch.Generator().manual_seed(args.random_seed)

        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [n_train, n_val],
            generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        print(f"Training set length: {len(train_dataset)}")
        print(f"Validation set length: {len(val_dataset)}")
    else:
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = None
        print(f"Training set length: {len(full_train_dataset)}")

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
    elif args.g_model == "willbenamed":
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
    # test_dataset = TensorDataset(test_tensor, test_labels)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    # Prepare inputs from original test indices
    y_test, y_test_full = [], []
    test_indices = np.random
    for file_test in files_test:
        record = wfdb.rdrecord(file_test)
        signal = record.p_signal[:, 0] 
        
        for start_idx in indices_ts_test:
            ts = signal[start_idx : start_idx + args.n]

            ts_fixed = normalize(ts)
        
            y_test_full.append(ts_fixed)

    # y_test_full = np.concatenate(y_test, axis=0) 
    X_test_full = np.stack([
        MP_compute_single(
                y_test_full[i], m,
                norm=args.enable_mp_norm,
                mpd_only=args.enable_mpd_only,
                znorm=args.znorm_mp,
                embedding=args.enable_mp_embedding,
                fill_value=args.fill_value
            )
            for i in range(len(y_test_full))
        ])
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
    #fake_data = normalize(fake_data)
    test_file_names = ["ecg_"+str(i) for i in range(len(X_test_full))]
    # Plot results
    if args.plot:
        plot_res(model_save_path, test_labels, fake_data, test_file_names, args.m, args.znorm_mp)
    
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

    save_args(args, model_save_path, "config.json")

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
    