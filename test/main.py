# GLOBAL IMPORTS
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os, argparse, glob, json
import stumpy
from scipy import stats

# LOCAL IMPORTS
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_path)  # Add root directory to path

from src.training.objectives import objective_function_exponential_pytorch, objective_function_pytorch
from src.utils_matrix_profile import compute_matrix_profile_distance
from model import Generator
from train import train_g

def normalize(time_series : np.ndarray) -> np.ndarray:
    rng = time_series.max() - time_series.min()
    if rng == 0:
        return np.zeros_like(time_series, dtype=float)
    return (time_series - time_series.min()) / rng

def pearson_correlation(x, y):
    """Compute the Pearson correlation between two time series x and y."""
    if np.all(x == y) or np.all(x == -y):
        return 1.0
    elif np.all(x == x[0]) or np.all(y == y[0]):
        return np.NaN
    return np.abs(stats.pearsonr(x, y).statistic)

def pearson_r2(y_pred, y_true, eps=1e-8):
    """
    Loss = 1 - r^2
    where r is the Pearson correlation between y_pred and y_true.

    Parameters
    ----------
    y_pred : list or np.ndarray
        Predicted time series
    y_true : list or np.ndarray
        Ground-truth time series
    eps : float
        Small constant for numerical stability

    Returns
    -------
    loss : float
        1 - Pearson correlation squared
    """

    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    # Center
    vx = y_pred - y_pred.mean()
    vy = y_true - y_true.mean()

    # Pearson correlation
    r_num = np.sum(vx * vy)
    r_den = np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + eps
    r = r_num / r_den

    return r ** 2


def plot_res(folder_name, real_data, fake_data, utility_distance, name_list, m, znorm, plot=True):
    original_color = 'tab:blue'
    fake_color = 'tab:red'
    for real_ts, fake_ts, file_name, utility_loss in zip(real_data, fake_data, name_list, utility_distance):
        file_name_short = os.path.splitext(os.path.basename(file_name))[0]
        real_ts = real_ts.numpy().squeeze().astype(np.float64)
        fake_ts = fake_ts.numpy().squeeze().astype(np.float64)
        fake_ts = normalize(fake_ts)
        os.makedirs(os.path.join(folder_name, file_name_short), exist_ok=True)
        res_json = {
            "data": real_ts.tolist(),
            "fake_data": fake_ts.tolist(),
            "utility_loss": float(utility_loss)
        }

        with open(os.path.join(folder_name, file_name_short, "results.json"), "w") as f:
            json.dump(res_json, f, indent=4)
        
        if plot:
            mp_real = stumpy.stump(real_ts, m=m, normalize=znorm)
            mp_fake = stumpy.stump(fake_ts, m=m, normalize=znorm)
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
    print(f"Results saved at {folder_name}")

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
    parser.add_argument("-obj_func", type=str, default="default", help="Define the objective function used in the training")
    parser.add_argument("-alpha", type=float, default=0.05, help="Define the parameter used in exponential objective function")
    parser.add_argument("-train_ratio", type=float, default=0.01, help="Training dataset ratio")
    args = parser.parse_args()

    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
    train_epoch = int(args.e)

    category = args.category
    ### The values need to be normalized !!!
    if category == "theoretical":
        # 1. Load synthetic data
        torch.manual_seed(args.random_seed)
        # B: batch size (number of users), 
        # T: time step (length of each time series), 
        # D: dimension (Number of variables per time step)
        X_full = torch.rand(args.n_ts, args.n, 1)
    
    elif category == "ecg":
        ecg_dir = "data/ecg/original"
        files = sorted(glob.glob(os.path.join(ecg_dir, "ecg_*.npy")))

        if len(files) < args.n_ts:
            raise ValueError(f"Not enough ECG files: found {len(files)}, but need {args.n_ts}")

        all_series = []
        for i in range(args.n_ts):
            ts = np.load(files[i]).astype(np.float32)  # Ensure float32
            if ts.ndim == 1:
                ts = np.expand_dims(ts, axis=-1)  # Convert (T,) → (T, 1)
            all_series.append(torch.tensor(ts))  # Each is (T, 1)

        # Optional: truncate or pad to fixed length args.n
        fixed_length_series = []
        for ts in all_series:
            T = ts.shape[0]
            if T >= args.n:
                ts_fixed = ts[:args.n]
            else:
                padding = torch.zeros(args.n - T, ts.shape[1])
                ts_fixed = torch.cat([ts, padding], dim=0)
            fixed_length_series.append(normalize(ts_fixed))

        # Stack all into (n_ts, n, 1)
        X_full = torch.stack(fixed_length_series)
    
    elif category == "energy":
        energy_dir = "data/energy/original"
        files = sorted(glob.glob(os.path.join(energy_dir, "energy_*.npy")))

        if len(files) < args.n_ts:
            raise ValueError(f"Not enough energy files: found {len(files)}, but need {args.n_ts}")

        all_series = []
        for i in range(args.n_ts):
            ts = np.load(files[i]).astype(np.float32)
            if ts.ndim == 1:
                ts = np.expand_dims(ts, axis=-1)  # Ensure shape (T, 1)
            all_series.append(torch.tensor(ts))  # Each: (T, 1)

        # Optional: truncate or pad to fixed length args.n
        fixed_length_series = []
        for ts in all_series:
            T = ts.shape[0]
            if T >= args.n:
                ts_fixed = ts[:args.n]
            else:
                padding = torch.zeros(args.n - T, ts.shape[1])
                ts_fixed = torch.cat([ts, padding], dim=0)
            fixed_length_series.append(normalize(ts_fixed))

        # Stack into (n_ts, n, 1)
        X_full = torch.stack(fixed_length_series)
    else:
        raise ValueError(f"Don't have such dataset name d {category}")

    # 2. Split: 50% train, 25% member test, 25% non-member, and store the file names
    torch.manual_seed(args.random_seed)
    n_ts = len(X_full)
    n_train = int(n_ts*args.train_ratio)
    train_set, test_set = random_split(X_full, [n_train, n_ts - n_train])

    # Convert Subset -> Tensor
    train_tensor = torch.stack([X_full[i] for i in train_set.indices])
    train_dataset = TensorDataset(train_tensor, torch.zeros(len(train_tensor)))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize model hyperparameters
    _, n, d = next(iter(train_loader))[0].shape  # n = time series length, d = 1
    hidden_dim = 64
    mp_dim = 2  # MPD + MPI

    # Models
    G = Generator(input_dim=mp_dim, hidden_dim=hidden_dim, output_length=n)

    # 3. Train G
    model_save_path = f"test/results/{time_str}/"
    if args.obj_func == "default":
        objective_func = objective_function_pytorch
    elif args.obj_func == "exponential":
        objective_func = objective_function_exponential_pytorch
    else:
        objective_func = objective_function_mpd_pytorch 
    G, g_loss_list = train_g(train_loader, G, device='cpu', checkpoint_path=model_save_path, epoch=train_epoch, mp_window_size=args.m, k_violation=args.k, alpha=args.alpha, objective_func=objective_func, input_dim=mp_dim )
    G.load_state_dict(torch.load(model_save_path+"best_model.pth"))
    

    # 4. Generate samples
    G.eval()
    B, T, D = next(iter(train_loader))[0].shape
    with torch.no_grad():
        mp_input_list = []
        for ts_tensor in test_set:
            ts = ts_tensor.squeeze().numpy().astype(np.float64)
            mp = stumpy.stump(ts, m=args.m)
            mp_dist = mp[:, 0].astype(np.float32)
            mp_index = mp[:, 1].astype(np.float32)  # even if it's originally int
            mp_clean = np.stack([mp_dist, mp_index], axis=1)
            mp_clean = np.nan_to_num(mp_clean, nan=0.0, posinf=0.0, neginf=0.0)
            mp_tensor = torch.tensor(mp_clean, dtype=torch.float32)
            mp_input_list.append(mp_tensor)

        mp_input_batch = torch.stack(mp_input_list).to('cpu')  # (B, n-m+1, 2)
        fake_data = G(mp_input_batch).cpu()  # Output: (B, n, 1)

    
    # Save results
    plot_res(model_save_path, test_set, fake_data, [files[i] for i in test_set.indices], args.m, plot=args.plot)
    
    # 5. Evaluate Utility: Matrix Profile
    utility_score = np.mean([compute_matrix_profile_distance(real_x.squeeze(), fake_x.squeeze()) for real_x, fake_x in zip(test_set, fake_data)])
    print(f"✅ Matrix Profile Distance (lower is better): {utility_score:.4f}")

    # Plot the first feature (D=1 assumed) or slice index 0
    plt.figure(figsize=(5, 12))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(fake_data[i, :, 0], label=f"Fake Sample {i}")
        plt.plot(normalize(test_set[i].numpy()[:, 0]), label=f"Real Sample {i}")
        plt.legend()
        plt.tight_layout()

    plt.suptitle("Generated Time Series (First Feature)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(model_save_path, "time_series.png"))
    plt.close()


    # 7. Plot the loss curves
   
    g_loss_list_np = np.array(g_loss_list)
    plt.figure(figsize=(8, 5))
    plt.plot(g_loss_list, label="Generator Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("G Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, "loss.png"))
    plt.close()

    save_args(args, model_save_path, "config.json")