import numpy as np
import stumpy
try:
    from numba import njit
    JIT = True
except Exception:
    JIT = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

def compute_matrix_profile_distance(real_series, fake_series, window_size=10, normalize=True):
    """
    Compute the average Euclidean matrix profile distance + average index mismatch
    between real and fake univariate time series sets.
    """
    real_np = real_series.numpy().astype(np.float64)
    fake_np = fake_series.numpy().astype(np.float64)

    total_dist = 0.0
    total_index_mismatch = 0.0
    
    mp_real = stumpy.stump(real_np, m=window_size, normalize=normalize)
    mp_fake = stumpy.stump(fake_np, m=window_size, normalize=normalize)

    # Distance profile
    dist = np.linalg.norm(mp_real[:, 0] - mp_fake[:, 0]) # L2 norm
    # Index mismatch
    index_diff = mp_real[:, 1] != mp_fake[:, 1]
    mismatch = index_diff.sum() / len(index_diff)
    total_dist += dist
    total_index_mismatch += mismatch

    return total_dist + total_index_mismatch

def MP_compute_recursive(ts_data, m, norm=False, dim=2, znorm=True):
    """
    Compute MP from the time series from ts_data vector, return list dimension [n_ts, 2, n-m+1]
    """ 
    mp_list = []
    for ts in ts_data:
        ts = np.array(ts, dtype=np.float64)

        profile = stumpy.stump(ts, m=m, normalize=normalize)
        if norm:
            mpd, mpi = normalized_MP(profile)
        else:
            mpd = profile[:, 0].astype(np.float32)
            mpi = profile[:, 1].astype(int)
        if dim==2:
            mp_list.append([mpd, mpi])
        else:
            mp_list.append(mpd)
    return np.array(mp_list, dtype=np.float32)

def znormalized_MP(mp):
    mpd = mp[:, 0].astype(np.float32)
    mpi = mp[:, 1].astype(int)
    L = len(mpd)
    dist_mean = np.mean(mpd)
    dist_std  = np.std(mpd)
    mpd_norm = (mpd - dist_mean) / dist_std

    # --- normalize index channel to [-1, 1] ---
    mpi_norm = (mpi / (L - 1)) * 2 - 1
    return mpd_norm, mpi_norm

def normalized_MP(mp):
    mpd = mp[:, 0].astype(np.float32)
    mpi = mp[:, 1].astype(int)
    
    mpd_norm = normalize(mpd)
    mpi_norm = normalize(mpi)
    
    return mpd_norm, mpi_norm

def normalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())