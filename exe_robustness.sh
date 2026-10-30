# usage: bash exe_robustness.sh [dataset] [n_runs]   (default: ptbxl 5)
DATASET=${1:-ptbxl}
N_RUNS=${2:-5}   # runs per eps, appended to the previous ones in eval_metrics.json
case "${DATASET}" in
    ptbxl|arrhythmia|arrhythmia_xl|ltdb|ptb_diagnostic) CATEGORY=ecg ;;
    tdbrain|eegmmidb) CATEGORY=eeg ;;
    *) echo "Unknown dataset '${DATASET}' (choose from: ptbxl arrhythmia arrhythmia_xl ltdb ptb_diagnostic tdbrain eegmmidb)"; exit 1 ;;
esac

# EPS_LIST=(100.0 1000.0 10000.0 100000.0 500000.0 800000.0 1000000.0 100000000.0)
EPS_LIST=(100.0 1000.0 10000.0 100000.0 500000.0 800000.0 1000000.0 100000000.0)
# EPS_LIST=(100000.0)
# only one component perturbed at a time: first the MPD, then the MPI
for rob in mpd mpi; do
    for eps in "${EPS_LIST[@]}"; do
        for run in $(seq 1 ${N_RUNS}); do
            echo "=== dataset=${DATASET} rob=${rob} eps=${eps} run ${run}/${N_RUNS} ==="
            python3 test/eval_robustness.py \
                -n_ts 200 \
                -n 500 \
                -m 100 \
                -c ${CATEGORY} \
                -r 2026 \
                -g_model deepmp \
                -mp_embedding \
                -znorm \
                -fill 100.0 \
                -dataset ${DATASET} \
                --rob ${rob} \
                --eps_mpi ${eps} \
                --eps_mpd ${eps}
        done
    done
done
