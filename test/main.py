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