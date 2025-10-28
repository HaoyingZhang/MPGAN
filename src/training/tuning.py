import optuna, os, shutil, numpy as np, torch, random
from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime
import sys, argparse, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # Add root directory to path
from models.CNN import Discriminator as D_WillBeNamed
from models.WillBeNamed import Generator as G_WillBeNamed
from training.objectives import objective_function_unified
from utils_matrix_profile import compute_matrix_profile_distance, MP_compute_recursive
from training.train_baseline import train_gan

def normalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def build_models(n, m, hidden_dim=64, latent=False): 
    G = G_WillBeNamed(
        n=n,
        m=m,
        mp_channels=2,
        base_channels=64,
        num_blocks=6,
        dilations=(1,2,4,8,16,32),
        use_attention=True,
        z_dim=64 if latent else None,          # or None to disable latent z
        y_dim=None         # or e.g. 10 if you have labels/classes
        )
    D = D_WillBeNamed(input_dim=1, hidden_dim=hidden_dim)
    return G, D

@torch.no_grad()
def eval_generator_mp(
    G,
    val_loader,
    device="cpu",
    d_model="lstm",
    latent=True,
    m_eval=32,              # FIXED across trials
    alpha_eval=0.5,        # FIXED
    k_eval=1.0,             # FIXED
    activ_eval="relu",       # FIXED
    mp_m = 10
):
    G.eval()
    mp_vals = []
    for mp_input_batch, _real_ts in val_loader:
        mp_input_batch = mp_input_batch.to(device)
        if latent and getattr(G, "z_dim", None):
            z = torch.randn(mp_input_batch.size(0), 64, device=device)
            fake = G(mp_input_batch, z=z)
        else:
            fake = G(mp_input_batch)

        # Evaluate ONLY the distance term, with fixed settings
        mp_loss_val = objective_function_unified(
            x_list=fake.squeeze(-1),
            mp_list=mp_input_batch,
            m=mp_m,
            coeff_dist=1.0,          # fixed
            coeff_identity=1.0,      # fixed
            k=1.0,
            device=device,
            alpha=alpha_eval,
            identity_activation=activ_eval
        )
        mp_vals.append(mp_loss_val.item())

    return float(np.mean(mp_vals)) if mp_vals else float("inf")

def objective(trial, train_loader, device="cuda", base_checkpoint_dir="optuna_ckpts", n=200, m=10):
    # --- search space ---
    # Constants
    m = m
    n = n
    alpha = 0.5
    d_model = "lstm"
    k_violation    = trial.suggest_float("k_violation", 0.1, 1.0)
    activ_func     = trial.suggest_categorical("activ_func", ["relu", "exp"])
    latent         = trial.suggest_categorical("latent", [False, True])

    pi_mp          = trial.suggest_float("pi_mp", 1e-3, 1.0, log=True)
    pi_adv         = trial.suggest_float("pi_adv", 0.1, 5.0, log=True)
    coeff_dist     = trial.suggest_float("coeff_dist", 0.1, 5.0, log=True)
    coeff_identity = trial.suggest_float("coeff_identity", 0.1, 5.0, log=True)

    lr_G           = trial.suggest_float("lr_G", 1e-5, 5e-4, log=True)
    lr_D           = trial.suggest_float("lr_D", 1e-5, 5e-4, log=True)

    # You can also tune epoch/time_limit to speed search; keep modest for trials
    epoch          = trial.suggest_int("epoch", 5, 20)
    time_limit     = trial.suggest_int("time_limit", 15, 60)

    # --- per-trial checkpoint dir (clean slate each time) ---
    ckpt_dir = os.path.join(base_checkpoint_dir, f"trial_{trial.number}")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_global_seed(2025 + trial.number)

    # (re)build fresh models per trial
    G, D = build_models(n=args.n, m=args.m, latent=latent)

    # --- train ---
    try:
        _, _, D_loss, G_loss, g_adv_loss, mp_loss = train_gan(
            train_loader=train_loader,
            G=G,
            D_net=D,
            device=device,
            checkpoint_path=ckpt_dir,
            epoch=epoch,
            mp_window_size=args.m,
            k_violation=k_violation,
            alpha=alpha,
            activ_func=activ_func,
            time_limit=time_limit,
            d_model=d_model,
            latent=latent,
            pi_mp=pi_mp,
            pi_adv=pi_adv,
            coeff_dist=coeff_dist,
            coeff_identity=coeff_identity,
            lr_G=lr_G,
            lr_D=lr_D
        )
    except RuntimeError as e:
        # e.g., OOM → tell Optuna this config is bad
        raise optuna.exceptions.TrialPruned() from e

    # --- objective: minimize mp loss ---
    best_G = build_models(n, m, latent=latent)[0]  # fresh G
    best_G.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=device))
    best_G.to(device)
    val_score = eval_generator_mp(
        best_G, train_loader,
        device="cpu",
        d_model=d_model,
        latent=latent,
        m_eval=10,          # keep fixed across all trials
        alpha_eval=0.5,
        k_eval=1.0,
        activ_eval="relu",
        mp_m=m
    )
    # report & (optional) prune based on intermediate value
    trial.report(val_score, step=epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_score  # MINIMIZE this (lower MP distance is better)

def run_optuna(train_loader, n, m, n_trials=30, device="cuda"):
    # Sampler/Pruner choices you can tweak
    sampler = optuna.samplers.TPESampler(seed=123)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"gan_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    study.optimize(
        lambda t: objective(t, train_loader=train_loader, device=device, n=n, m=m),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=True
    )

    print("Best value:", study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='===== GAN to generate synthetic time series with the given Matrix Profile =====')
    parser.add_argument("-n_ts", type=int, required=True, help="Length of the dataset")
    parser.add_argument("-n", type=int, required=True, help="Length of the considered time series")
    parser.add_argument("-m", type=int, required=True, help="Subsequence length for the Matrix Profile")
    parser.add_argument("-r", "--random_seed", type=int, default=None, help="Random seed used to generate the time series (default: None)")
    parser.add_argument("-c", "--category", type=str, default="theoretical", help="Category of the time series: 'theoretical', 'energy', or 'ecg' (default: 'theoretical')")

    args = parser.parse_args()
    m = args.m
    n = args.n
    if n-m+1 <= 0:
        raise ValueError(f"Need n - m + 1 > 0, got n={n}, m={m}")

    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    y_full = []

    category = args.category
    if args.category == "theoretical":
        rng = np.random.default_rng(args.random_seed)
        for i in range(args.n_ts):
            ts = rng.random(n).astype(np.float32)
            y_full.append(normalize(ts))       # use your normalize
    else:
        if args.category == "ecg":
            data_dir = "data/ecg/ecg_long"
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
    X_full = MP_compute_recursive(y_full, m)

    # 2. Split: 60% train, 40% test
    generator = torch.Generator().manual_seed(args.random_seed)
    n_ts = len(X_full)
    train_set, test_set = random_split(X_full, [10*n_ts//14, 4*n_ts//14], generator=generator)
    print(f"Training set length: {len(train_set.indices)}")
    print(f"Test set length: {len(test_set.indices)}")

    # Convert Subset -> Tensor

    # Train and test tensors
    train_tensor = torch.stack([torch.tensor(X_full[i], dtype=torch.float32) 
                                for i in train_set.indices])
    train_tensor = train_tensor.view(train_tensor.size(0), -1)

    test_tensor = torch.stack([torch.tensor(X_full[i], dtype=torch.float32) 
                                for i in test_set.indices])
    test_tensor = test_tensor.view(test_tensor.size(0), -1)

    # Train and test labels
    train_labels = torch.stack([torch.tensor(y_full[i], dtype=torch.float32) 
                                for i in train_set.indices])
    test_labels = torch.stack([torch.tensor(y_full[i], dtype=torch.float32) 
                                for i in train_set.indices])

    train_dataset = TensorDataset(train_tensor, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize model hyperparameters
    batch_size, n_input = next(iter(train_loader))[0].shape  # n_input = 2*(n-m+1)

    assert n_input == 2*(n-m+1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    study = run_optuna(train_loader, n_trials=40, device=device, n=args.n, m=args.m)

    # Rebuild models with best params and retrain longer if you want a final model:
    # best = study.best_trial.params
    # G, D = build_models()
    # _ = train_gan(
    #     train_loader,
    #     G, D,
    #     device="cuda",
    #     checkpoint_path="best_final",
    #     epoch=50,                 # longer training for final
    #     mp_window_size=best["mp_window_size"],
    #     k_violation=best["k_violation"],
    #     alpha=best["alpha"],
    #     activ_func=best["activ_func"],
    #     time_limit=120,           # bigger budget now
    #     d_model=best["d_model"],
    #     latent=best["latent"],
    #     pi_mp=best["pi_mp"],
    #     pi_adv=best["pi_adv"],
    #     coeff_dist=best["coeff_dist"],
    #     coeff_identity=best["coeff_identity"],
    #     lr_G=best["lr_G"],
    #     lr_D=best["lr_D"]
    # )