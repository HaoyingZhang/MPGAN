#!/bin/bash
# Environment for MPGAN, including the IPOPT reconstruction (test/eval_robustness.py --ipopt,
# test/eval_utility.py --ipopt). IPOPT (pyoptsparse + cyipopt) is only available from conda-forge,
# so everything is installed in one conda environment.
#
# usage: bash env.bash [env_name]   (default: mpgan), then: conda activate mpgan
set -e  # stop on first error

ENV_NAME=${1:-mpgan}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# If gcc is missing (needed to build some pip packages):
# sudo apt update -y && sudo apt install -y build-essential python3-dev

eval "$(conda shell.bash hook)"
conda create -y -n "${ENV_NAME}" python=3.10
conda activate "${ENV_NAME}"
# Ignore the packages of ~/.local (pip would skip them as already installed, and old versions,
# e.g. stumpy needing pkg_resources, would shadow the ones of the environment)
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1

# IPOPT solver (attack_matrix_profile/src/rmpi)
conda install -y -c conda-forge pyoptsparse mpi4py cyipopt

# MPGAN
pip install --no-cache-dir -r "${ROOT}/requirements.txt"
pip install --no-cache-dir scipy scikit-learn xgboost numba joblib h5py wfdb mne PyWavelets \
    tsfresh tslearn dtaidistance opendp python-dateutil

# Other packages imported by the IPOPT solver
pip install --no-cache-dir psutil sqlitedict ucimlrepo pyyaml

# Check that IPOPT is available
python -c "import stumpy, h5py, tsfresh; from pyoptsparse import OPT; OPT('IPOPT'); print('IPOPT ok')"

# If the check fails with a GLIBCXX / libstdc++ error, use the conda libstdc++:
# conda env config vars set LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
# conda deactivate && conda activate "${ENV_NAME}"

echo "Done. Activate the environment with: conda activate ${ENV_NAME}"
