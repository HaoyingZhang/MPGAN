import stumpy
import numpy as np
from utils import pearson_correlation
import os
import matplotlib.pyplot as plt

def plot_res(fake_ts, ts_original, m, folder_name, file_name_short, znorm=True, original_color='tab:blue', fake_color='tab:red'):
    """
    Plot and save the result of inversion.
    
    :param fake_ts: The invert time series after normalization
    :param ts_original: The original time series (ground truth)
    :param m: The subsequence length
    :param folder_name: The folder name to save the results
    :param file_name_short: The file name to save the results (ex: ecg_0)
    :param znorm: Set to true if using z-normalized matrix profile
    :param original_color: The color used to plot the original time series
    :param fake_color: The color used to plot the invert time series
    """
    mp_real_clean = mp_real = stumpy.stump(ts_original, m=m, normalize=znorm)
    mp_fake = stumpy.stump(fake_ts, m=m, normalize=znorm)
    mp_real_clean = mp_real[:, [0, 1]].astype(np.float32)
    mp_real_clean = np.nan_to_num(mp_real_clean, nan=0.0, posinf=0.0, neginf=0.0)
    mp_fake_clean = mp_fake[:, [0, 1]].astype(np.float32)
    mp_fake_clean = np.nan_to_num(mp_fake_clean, nan=0.0, posinf=0.0, neginf=0.0)
    figsize=(19.2,9)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, squeeze=False)

    # Plot the time series
    axs[0, 0].plot(ts_original, label='Original TS', color=original_color)
    axs[0, 0].plot(fake_ts, label='Fake TS', color=fake_color)
    axs[0, 0].set_title(f"Pearson Corr: {round(pearson_correlation(ts_original, np.array(fake_ts)), 2)}", fontsize=12)
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