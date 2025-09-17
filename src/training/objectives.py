import torch
import torch.nn.functional as F


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
    # mp = mp_list[0]
    if k < 1.00:
        # Take top k violations
        violation_k = min(int(mp.shape[0] * k), mp.shape[0] - 1)
        violation = True
    for x, mp in zip(x_list, mp_list):
        x = x.to(device)
        mp = mp.to(device)
        n_mp = mp.shape[0]/2

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
        

        total_loss += (coeff_dist * distance_loss + coeff_identity * identity_loss) / n_mp

    return total_loss / len(x_list)

def objective_function_exponential_pytorch (x_list, mp_list, m, coeff_dist=1.0, coeff_identity=1.0, device='cuda', alpha = 0.05, k = None):
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
    mp = mp_list[0]
    assert mp.shape[0] == 2*(x_list[0].shape[0]-m+1)

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

        total_loss += (coeff_dist * distance_loss + coeff_identity * identity_loss ) / n_mp

    return total_loss / len(x_list)