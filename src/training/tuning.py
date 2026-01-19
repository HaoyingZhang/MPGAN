import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # Add root directory to path
from src.dataloader import MemmapDataset
from src.models.WillBeNamed import Generator as G_WillBeNamed
from src.training.train_baseline import train_inverse

# Assuming your classes and functions are in your local path
# from model_file import G_WillBeNamed, train_inverse, MemmapDataset, normalize, MP_compute_single

def objective(trial):
    # 1. Suggest Hyperparameters
    # Learning rates and regularizations
    lr_g = trial.suggest_float("lr_g", 1e-6, 1e-3, log=True)
    p_drop = trial.suggest_float("p_drop", 0.1, 0.5)
    k = trial.suggest_float("k", 0.1, 0.9)
    # Loss coefficients (The "Pi" values)
    obj_func = trial.suggest_categorical("obj_func", ["exp", "relu"])
    fill_value = trial.suggest_int("fill_value", 100,1000, log=True)
    
    # Model Architecture
    base_channels = trial.suggest_categorical("base_channels", [32, 64, 128])
    num_blocks = trial.suggest_int("num_blocks", 4, 8)
    # Match dilations to num_blocks
    dilations = tuple([2**i for i in range(num_blocks)])

    # 2. Setup Data 
    n, m = 200, 10 
    L = n - m + 1
    C = 2 
    batch_size = 64
    
    train_dataset = MemmapDataset(
        "X_train.dat", "y_train.dat", 
        shape_X=(400, L, C), shape_y=(400, n)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    G = G_WillBeNamed(
        n=n, m=m,
        mp_channels=L,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dilations=dilations,
        use_attention=True,
        z_dim=None,
        p_drop=p_drop,
        dropout=True
    ).to(device)

    # 4. Train
    
    try:
        _, _, _, _,best_g_loss = train_inverse(
            train_loader,
            None, # Val loader
            G,
            device=device,
            epoch=5,
            mp_window_size=m, 
            k_violation=k, 
            alpha=0.5, 
            activ_func=obj_func, 
            time_limit=60,
            pi_mp=0.01,
            pi_mse=1.0,
            pi_pcc=1.0,
            pi_grad=1.0,
            lr_G=lr_g,
            latent=False,
            coeff_dist = 0.8,
            coeff_identity=0.2,
            mp_norm=True,
            embedding_mp=True,
            fill_value=fill_value
        )


    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0 # Return poor score on failure

    # We want to MAXIMIZE Pearson Correlation Coefficient
    return 1/best_g_loss

if __name__ == "__main__":
    # Create a study to maximize correlation
    study = optuna.create_study(direction="maximize")
    
    # You can run this for 50-100 trials
    study.optimize(objective, n_trials=50)

    print("\nBest Trial:")
    trial = study.best_trial
    print(f"  Value (PCC): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv")