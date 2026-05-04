# Privacy Evaluation Framework for Matrix Profile

A framework for empirically evaluating the **reconstruction risk** and **re-identification risk** of time series data de-identified via Matrix Profile-based methods.

---

Two attack pipelines are implemented:

| Attack Type | Question | Metric |
|---|---|---|
| **Reconstruction** | Can the original signal and sensitive features be recovered from the de-identified version? | RMSE, correlation, Relative Error |
| **Re-identification** | Can the source subject be identified from the de-identified signal? | Re-identification Rate (RIR) |

Tunning the parameters:

python3 src/training/tuning.py -n_ts 140 -n 1000 -m 10 -r 2025 -c ecg

Post-processing:

test/post_processing.py

