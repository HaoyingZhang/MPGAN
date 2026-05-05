#!/bin/bash

#SBATCH -c 10                           # 20 cpu
#SBATCH --job-name=MPGAN      # Job name
#SBATCH --time=48:00:00                   # Maximum runtime
#SBATCH --mem=20G
#SBATCH -p GPU                       # Compute partition


#SBATCH --export=ALL                     # Export all environment variables
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-best

# === Job execution steps ===
# Load Bash configuration (needed for Conda)
# source ~/.bashrc

# Activate Conda environment
# conda activate mpgan

python3 test/eval.py \
    -n_ts 200 \
    -n 500 \
    -m 100 \
    -c eeg \
    -dataset tdbrain \
    -train_id 0 100 \
    -test_id 1000 1200 \
    -e 10 \
    -r 2026 \
    -k 0.60 \
    -g_model deepmp \
    -obj_func exp \
    -alpha 0.5 \
    -pi_mp 0.000000 \
    -pi_mse 1.0 \
    -pi_pcc 1.0 \
    -pi_grad 1.0 \
    -lr_g 0.0008 \
    -coeff_dist 0.8 \
    -coeff_index 0.2 \
    -mp_embedding \
    -znorm \
    -time 10 \
    -fill 100.0 \
    -p \
    -test


echo "✅ Job completed successfully!"

conda deactivate