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

def compute_matrix_profile_distance(real_series, fake_series, window_size=10):
    """
    Compute the average Euclidean matrix profile distance + average index mismatch
    between real and fake univariate time series sets.
    """
    real_np = real_series.numpy().astype(np.float64)
    fake_np = fake_series.numpy().astype(np.float64)

    total_dist = 0.0
    total_index_mismatch = 0.0
    
    mp_real = stumpy.stump(real_np, m=window_size)
    mp_fake = stumpy.stump(fake_np, m=window_size)

    # Distance profile
    dist = np.linalg.norm(mp_real[:, 0] - mp_fake[:, 0])
    # Index mismatch
    index_diff = mp_real[:, 1] != mp_fake[:, 1]
    mismatch = index_diff.sum() / len(index_diff)
    total_dist += dist
    total_index_mismatch += mismatch

    return total_dist + total_index_mismatch

def MP_compute_recursive(ts_data, m):
    """
    Compute MP from the time series from ts_data vector, return list dimension [n_ts, 2, n-m+1]
    """ 
    mp_list = []
    for ts in ts_data:
        ts = np.array(ts, dtype=np.float64)

        profile = stumpy.stump(ts, m)

        mpd = profile[:, 0]
        mpi = profile[:, 1].astype(int)

        mp_list.append([mpd, mpi])
    return np.array(mp_list, dtype=np.float32)


