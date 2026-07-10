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

python3 test/eval_robustness.py \
    -n_ts 200 \
    -n 500 \
    -m 100 \
    -c ecg \
    -r 2026 \
    -g_model deepmp \
    -mp_embedding \
    -znorm \
    -fill 100.0 \
    -dataset ptbxl\
    --rob mpd \
    --eps 1000 \


echo "✅ Job completed successfully!"

conda deactivate