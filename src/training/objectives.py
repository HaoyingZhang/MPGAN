import torch
import torch.nn.functional as F
import math
import numpy as np


def znormalized(x, eps=1e-8):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True) + eps
    return (x - mean) / std


def compute_znormalized_distance_matrix(x, m):
    """
    Computes a full pairwise z-normalized Euclidean distance matrix between all subsequences of length m from x.
    Args:
        x: (T,) 1D tensor
        m: int, length of subsequences
    Returns:
        D: (n_subseq, n_subseq) tensor of pairwise distances
    """
    T = x.shape[0]
    n_subseq = T - m + 1
    subsequences = x.unfold(0, m, 1)  # shape: (n_subseq, m)
    subsequences = znormalized(subsequences)

    # Compute pairwise Euclidean distances
    diff = subsequences.unsqueeze(1) - subsequences.unsqueeze(0)  # (n_subseq, n_subseq, m)
    D = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # (n_subseq, n_subseq)
    return D

def objective_function_pytorch (x_list, mp_list, m, coeff_dist=1.0, coeff_identity=1.0, k=1.00, device='cuda', alpha = None):
    """
    Differentiable PyTorch version of matrix profile objective function with optimisation in identity violation.
    
    Args:
        x_list: list of 1D PyTorch tensors (time series) of shape (T,)
        mp_list: list of tensors of shape (n_subseq, 2), where mp[:, 0] = MPD, mp[:, 1] = MPI
        m: subsequence length
        coeff_dist, coeff_identity: weighting coefficients
        device: computation device
    Returns:
        total_loss: scalar PyTorch tensor
    """
    total_loss = 0.0
    violation = False
    mp_0 = mp_list[0]
    assert mp_0.shape[0] == 2*(x_list[0].shape[0]-m+1)
    if k < 1.00:
        # Take top k violations
        violation_k = min(int(mp_0.shape[0] * k), mp_0.shape[0] - 1)
        violation = True
    for x, mp in zip(x_list, mp_list):
        x = x.to(device)
        mp = mp.to(device)
        n_mp = int(mp.shape[0]/2)

        D = compute_znormalized_distance_matrix(x, m)  # shape: (n_mp, n_mp)

        mpd = mp[:n_mp]  # ground-truth distances
        mpi = mp[n_mp:].long()  # ground-truth indices

        # --- Distance Loss ---
        dist_est = D[torch.arange(n_mp), mpi]
        distance_loss = torch.sum((mpd - dist_est) ** 2)

        # --- Identity Loss ---
        if violation:
            violations = dist_est.unsqueeze(1) - D
            violations.fill_diagonal_(float('-inf'))
            topk_vals, _ = torch.topk(violations, k=violation_k, dim=1)
            identity_loss = torch.clamp(topk_vals, min=0.0).sum()
        else:
            identity_diff = torch.clamp(dist_est.unsqueeze(1) - D, min=0.0)  # shape: (n_mp, n_mp) clamp => RELU
            identity_diff.fill_diagonal_(0.0)  # zero self-comparisons
            identity_loss = torch.sum(identity_diff)
        

        total_loss += (coeff_dist * distance_loss + coeff_identity * identity_loss)

    return total_loss / len(x_list)

def objective_function_exponential_pytorch (x_list, mp_list, m, coeff_dist=1.0, coeff_identity=1.0, device='cuda', alpha = 0.05, k = 1.00):
    """
    Differentiable PyTorch version of matrix profile objective function with optimisation in identity violation.
    
    Args:
        x_list: list of 1D PyTorch tensors (time series) of shape (T,)
        mp_list: list of tensors of shape (n_subseq, 2), where mp[:, 0] = MPD, mp[:, 1] = MPI
        m: subsequence length
        coeff_dist, coeff_identity: weighting coefficients
        device: computation device
    Returns:
        total_loss: scalar PyTorch tensor
    """
    total_loss = 0.0
    mp_0 = mp_list[0]
    assert mp_0.shape[0] == 2*(x_list[0].shape[0]-m+1)

    for x, mp in zip(x_list, mp_list):
        x = x.to(device)
        mp = mp.to(device)
        n_mp = int(mp.shape[0]/2)

        D = compute_znormalized_distance_matrix(x, m)  # shape: (n_mp, n_mp)

        mpd = mp[:n_mp]  # ground-truth distances
        mpi = mp[n_mp:].long()  # ground-truth indices

        # --- Distance Loss ---
        dist_est = D[torch.arange(n_mp), mpi]
        distance_loss = torch.sum((mpd - dist_est) ** 2)

        # --- Identity Loss ---
        # Compute violations
        violations = dist_est.unsqueeze(1) - D  # shape: (n_mp, n_mp)

        # Mask out diagonal ± alpha*m
        band_size = int(alpha * m)
        mask = torch.ones_like(violations, dtype=torch.bool)
        for i in range(n_mp):
            start = max(0, i - band_size)
            end = min(n_mp, i + band_size + 1)
            mask[i, start:end] = False

        # Apply exponential penalty to masked entries only
        exp_penalty = torch.zeros_like(violations)
        exp_penalty[mask] = torch.exp(torch.clamp(violations[mask], min=0.0)) - 1.0

        # Sum all penalties
        identity_loss = exp_penalty.sum()

        total_loss += (coeff_dist * distance_loss + coeff_identity * identity_loss )

    return total_loss / len(x_list)


def objective_function_unified(
    x_list,
    mp_list,
    m,
    coeff_dist: float = 1.0,
    coeff_identity: float = 1.0,
    device: str = "cuda",
    identity_activation: str = "relu",  # "relu" or "exp" (for exp(x)-1)
    alpha: float | None = 0.5,         # if set, ignores band |i-j| <= alpha*m
    k: float = 1.0                      # if 0<k<1: keep top ceil(k*(n-1)) per row; else use all
):
    """
    Differentiable PyTorch matrix-profile objective with flexible identity penalty.

    - identity_activation="relu":     penalty = max(violation, 0)
    - identity_activation="exp":      penalty = exp(max(violation,0)) - 1  (via torch.expm1)

    - alpha: ignore matches within a diagonal band of width floor(alpha*m)
    - k in (0,1): per row keep top ceil(k*(n-1)) positive violations only; otherwise sum all

    Shapes:
      x:  (T,)
      mp: (2n,), first n are MPD, last n are MPI (ints), where n = T - m + 1
    """
    assert len(x_list) == len(mp_list)
    total_loss = 0

    # For k logic we need n; validate shapes using the first pair
    mp0 = mp_list[0]
    assert mp0.shape[0] == 2 * (x_list[0].shape[0] - m + 1), "mp shape mismatch with x and m"

    # Precompute band size from alpha (if any)
    band_size = None
    if alpha is not None and alpha > 0:
        band_size = int(alpha * m)

    use_topk = (k is not None) and (0.0 < k < 1.0)

    for x, mp in zip(x_list, mp_list):
        x = x.to(device)
        mp = mp.to(device)

        L = (mp.shape[0] // 2)
        D = compute_znormalized_distance_matrix(x, m)  # (n,n)

        mpd = mp[:L]                 # ground-truth distances (float)
        mpi = mp[L:].long()          # ground-truth indices (int)

        # --- Distance loss: squared error on matched pairs ---
        dist_est = D[torch.arange(L, device=device), mpi]
        distance_loss = torch.sum((mpd - dist_est) ** 2)

        # --- Identity loss: penalize any j where D[i,j] < D[i,mpi[i]] (i.e., violation > 0) ---
        violations = dist_est.unsqueeze(1) - D  # (L,L) >0 means identity is violated

        # Exclude diagonal and (optionally) a ±alpha*m band efficiently
        if band_size is not None:
            i = torch.arange(L, device=device).unsqueeze(1)  # (n,1)
            j = torch.arange(L, device=device).unsqueeze(0)  # (1,n)
            band_mask = (j - i).abs() <= band_size          # True where inside band
        else:
            band_mask = torch.eye(L, dtype=torch.bool, device=device)  # only diagonal

        # Keep only positive violations outside the mask
        positive = torch.clamp(violations, min=0.0)
        positive = positive.masked_fill(band_mask, 0.0)

        # Activation choice
        if identity_activation.lower() == "relu":
            contrib = positive
        elif identity_activation.lower() == "exp":
            # exp(x)-1 but numerically stable and with meaningful gradient at 0
            contrib = torch.expm1(positive)
        else:
            raise ValueError("identity_activation must be 'relu' or 'exp'")

        # Top-k per row (after activation), or sum all
        if use_topk:
            k_count = max(1, min(L - 1, math.ceil(k * (L - 1))))
            topk_vals, _ = torch.topk(contrib, k=k_count, dim=1)
            identity_loss = topk_vals.sum()
        else:
            identity_loss = contrib.sum()

        total_loss = total_loss + (coeff_dist * distance_loss + coeff_identity * identity_loss)

    return total_loss/len(x_list)