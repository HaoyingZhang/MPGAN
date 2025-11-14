# GLOBAL IMPORTS
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os, argparse, glob, json
os.environ["NUMBA_DISABLE_CUDA"] = "1"
import stumpy
from scipy import stats

# LOCAL IMPORTS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # Add root directory to path
from models.LSTM import Generator as G_LSTM
from models.BiLSTM_latent import Generator as G_biLSTM
# from models.LSTM import Discriminator
# from models.BiLSTM import Pulse2pulseDiscriminator
from models.CNN import Discriminator as D
from models.CNN import Generator as G_CNN
from models.WillBeNamed import Generator as G_WillBeNamed
from training.train_baseline import train_gan, train_wgan_gp, train_inverse
from src.utils_matrix_profile import compute_matrix_profile_distance, MP_compute_recursive
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

def plot_res(folder_name, real_data, fake_data, name_list, m):
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

        mp_real = stumpy.stump(real_ts, m)
        mp_fake = stumpy.stump(fake_ts, m)
        mp_real_clean = mp_real[:, [0, 1]].astype(np.float32)
        mp_real_clean = np.nan_to_num(mp_real_clean, nan=0.0, posinf=0.0, neginf=0.0)
        mp_fake_clean = mp_fake[:, [0, 1]].astype(np.float32)
        mp_fake_clean = np.nan_to_num(mp_fake_clean, nan=0.0, posinf=0.0, neginf=0.0)
        figsize=(19.2,9)
        fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, squeeze=False)

        # Plot the time series
        axs[0, 0].plot(real_ts, label='Original TS', color=original_color)
        axs[0, 0].plot(fake_ts, label='Fake TS', color=fake_color)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='===== GAN to generate synthetic time series with the given Matrix Profile =====')
    parser.add_argument("-n_ts", type=int, required=True, help="Length of the dataset")
    parser.add_argument("-n", type=int, required=True, help="Length of the considered time series")
    parser.add_argument("-m", type=int, required=True, help="Subsequence length for the Matrix Profile")
    parser.add_argument("-e", type=int, default = 10, help="Number of epoches to train the network")
    parser.add_argument("-r", "--random_seed", type=int, default=None, help="Random seed used to generate the time series (default: None)")
    parser.add_argument("-c", "--category", type=str, default="theoretical", help="Category of the time series: 'theoretical', 'energy', or 'ecg' (default: 'theoretical')")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the original time series and the solutions (default: False)")
    parser.add_argument("-k", type=float, default = 1.0, help="Focus on optimizing the top k percent small distance, default : the original objective function")
    parser.add_argument("-g_model", type=str, default = "lstm", help="Choose the G model")
    parser.add_argument("-d_model", type=str, default = "lstm", help="Choose the D model")
    parser.add_argument("-obj_func", type=str, default = "relu", help="Define the objective function used in the training")
    parser.add_argument("-alpha", type=float, default = 0.05, help="Define the parameter used in exponential objective function")
    parser.add_argument("-pi_mp", type=float, default = 0.05, help="Define the coefficient of the condition loss")
    parser.add_argument("-pi_adv", type=float, default = 0.05, help="Define the coefficient of the adversary loss")
    parser.add_argument("-pi_ts", type=float, default = 0.05, help="Define the coefficient of the original ts loss")
    parser.add_argument("-latent", "--enable_latent", action="store_true", help="Latent dimension")
    parser.add_argument("-mp_norm", "--enalbe_mp_norm", action="store_true", help="Enable normalized MP")
    parser.add_argument("-lr_g", type=float, default=1e-5, help="Learning rate for Generator")
    parser.add_argument("-lr_d", type=float, default=1e-5, help="Learning rate for Discriminator")
    parser.add_argument("-coeff_dist", type=float, default = 1.0, help="Define the coefficient of the distance loss in MP")
    parser.add_argument("-coeff_index", type=float, default = 1.0, help="Define the coefficient of the index loss in MP")
    parser.add_argument("-time", type=int, default = None, help="Time limit" )
    parser.add_argument("-inj_proj", "--enable_inj_proj", action="store_true", help="Using projection to expand features")
    args = parser.parse_args()

    if args.obj_func not in ["relu", "exp"]: 
        parser.error(f"Must choose objective function between relu or exp, but got {args.obj_func}")

    m = args.m
    n = args.n
    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
    train_epoch = int(args.e)

    y_full = []

    category = args.category
    if args.category == "theoretical":
        rng = np.random.default_rng(args.random_seed)
        for i in range(args.n_ts):
            ts = rng.random(n).astype(np.float32)
            y_full.append(normalize(ts))       # use your normalize
    else:
        if args.category == "ecg":
            data_dir = "data/ecg/long"
            files = sorted(glob.glob(os.path.join(data_dir, "ecg_*.npy")))
        elif args.category == "energy":
            data_dir = "data/energy/original"
            files = sorted(glob.glob(os.path.join(data_dir, "energy_*.npy")))
        else:
            raise ValueError(f"Unknown dataset category: {args.category}")

        if len(files) < args.n_ts:
            raise ValueError(f"Not enough files: found {len(files)}, need {args.n_ts}")

        for i in range(args.n_ts):
            ts = np.load(files[i])                     # (T,) or (T,D)
            ts = np.asarray(ts, dtype=np.float32)
            if ts.ndim == 2:                           # pick first channel if multivariate
                ts = ts[:, 0]
            T = ts.shape[0]
            if T >= n:
                ts_fixed = ts[:n]
            else:
                pad = np.zeros((n - T,), dtype=np.float32)
                ts_fixed = np.concatenate([ts, pad], axis=0)
            y_full.append(normalize(ts_fixed))         # use your normalize

    # Calculate MP from the time series to compose X_full, X_full should be with dimension [n_ts, 2, n-m+1]
    X_full = MP_compute_recursive(y_full, m, norm=args.enalbe_mp_norm)

    # 2. Split: 60% train, 40% test
    generator = torch.Generator().manual_seed(args.random_seed)
    n_ts = len(X_full)
    train_set, val_set, test_set = random_split(X_full, [10*n_ts//14, 2*n_ts//14, 2*n_ts//14], generator=generator)
    print(f"Training set length: {len(train_set.indices)}")
    print(f"Validation set length: {len(val_set.indices)}")
    print(f"Test set length: {len(test_set.indices)}")

    # Convert Subset -> Tensor

    # Train and test tensors
    train_tensor = torch.stack([torch.tensor(X_full[i], dtype=torch.float32) 
                                for i in train_set.indices])
    train_tensor = train_tensor.view(train_tensor.size(0), -1)

    val_tensor = torch.stack([torch.tensor(X_full[i], dtype=torch.float32) 
                                for i in val_set.indices])
    val_tensor = val_tensor.view(val_tensor.size(0), -1)

    test_tensor = torch.stack([torch.tensor(X_full[i], dtype=torch.float32) 
                                for i in test_set.indices])
    test_tensor = test_tensor.view(test_tensor.size(0), -1)

    # Train and test labels
    train_labels = torch.stack([torch.tensor(y_full[i], dtype=torch.float32) 
                                for i in train_set.indices])
    val_labels = torch.stack([torch.tensor(y_full[i], dtype=torch.float32) 
                                for i in val_set.indices])
    test_labels = torch.stack([torch.tensor(y_full[i], dtype=torch.float32) 
                                for i in train_set.indices])

    train_dataset = TensorDataset(train_tensor, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = TensorDataset(val_tensor, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    # Initialize model hyperparameters
    batch_size, n_input = next(iter(train_loader))[0].shape  # n_input = 2*(n-m+1)
    
    hidden_dim = 64
    mp_dim = 2  # MPD + MPI

    assert n_input == 2*(n-m+1)

    # Enable latent
    if args.enable_latent:
        latent_dim = 16
    else:
        latent_dim = None
    # Models
    if args.g_model == "lstm":
        G = G_LSTM(input_dim=mp_dim, hidden_dim=hidden_dim, output_length=n)
    elif args.g_model == "bi-lstm":
        G = G_biLSTM(input_dim=mp_dim, hidden_dim=hidden_dim, output_length=n, latent_dim=latent_dim)
    elif args.g_model == "cnn":
        G = G_CNN(n=n, m=m)
    elif args.g_model == "willbenamed":
        G = G_WillBeNamed(
                n=n,
                m=m,
                mp_channels=2,
                base_channels=64,
                num_blocks=6,
                dilations=(1,2,4,8,16,32),
                use_attention=True,
                z_dim=64 if args.enable_latent else None,
                y_dim=None,         # or e.g. 10 if you have labels/classes
                use_in_proj=args.enable_inj_proj
            )

    if args.d_model == "lstm":
        D_net = D(input_dim=1, hidden_dim=hidden_dim)
    elif args.d_model == "pulse2pulse":
        D_net = Pulse2pulseDiscriminator(num_channels=1)

    # 3. Train GAN
    model_save_path = f"src/results/baseline/{time_str}/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    # G, _, d_loss_list, g_loss_list, g_adv_loss_list, mp_loss_list = train_gan(train_loader, 
    #                                             G, D_net, 
    #                                             device=device, 
    #                                             checkpoint_path=model_save_path, 
    #                                             epoch=train_epoch, 
    #                                             mp_window_size=args.m, 
    #                                             k_violation=args.k, 
    #                                             alpha=args.alpha, 
    #                                             activ_func=args.obj_func, 
    #                                             time_limit=args.time,
    #                                             d_model = args.d_model,
    #                                             pi_mp=args.pi_mp,
    #                                             pi_adv=args.pi_adv,
    #                                             lr_D=args.lr_d,
    #                                             lr_G=args.lr_g,
    #                                             latent=args.enable_latent,
    #                                             coeff_dist = args.coeff_dist,
    #                                             coeff_identity=args.coeff_index)
    G, G_loss, MP_loss, TS_loss, best_g_loss = train_inverse(train_loader,
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
                                                 pi_ts=args.pi_ts,
                                                 lr_G=args.lr_g,
                                                 latent=args.enable_latent,
                                                 coeff_dist = args.coeff_dist,
                                                 coeff_identity=args.coeff_index)

    G.load_state_dict(torch.load(model_save_path+"best_model.pth"))
    G = G.cpu()
    # 4. Generate samples
    G.eval()
    # test_dataset = TensorDataset(test_tensor, test_labels)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    # Prepare inputs from original test indices

    
    with torch.no_grad():
        if args.enable_latent:
            z = torch.randn(test_tensor.size(0), 64, device='cpu') if G.z_dim else None
            fake_data = G(test_tensor, z=z)
        else:
            fake_data = G(test_tensor)
    fake_data = normalize(fake_data)
    test_file_names = [files[i] for i in test_set.indices]
    # Plot results
    if args.plot:
        plot_res(model_save_path, test_labels, fake_data, test_file_names, args.m)
    
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


    # 7. Plot the loss curves
    # d_loss_list_np = np.array(d_loss_list)
    g_loss_list_np = np.array(G_loss)
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

    save_args(args, model_save_path, "config.json")