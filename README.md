# Privacy Evaluation Framework for Matrix Profile

A framework for empirically evaluating the **reconstruction risk** and **re-identification risk** of time series data de-identified via Matrix Profile-based methods.

---

Two attack pipelines are implemented:

| Attack Type | Question | Metric |
|---|---|---|
| **Reconstruction** | Can the original signal and sensitive features be recovered from the de-identified version? | RMSE, correlation, Relative Error |
| **Re-identification** | Can the source subject be identified from the de-identified signal? | Re-identification Rate (RIR) |

---

Data preparation:

PTBXL: Download from [This Link](https://physionet.org/content/ptb-xl/1.0.3/) and place the folder physionet.org under `data/`

Arrhythmia: Download the dataset from [This Link](https://physionet.org/content/mitdb/1.0.0/) and place the `1.0.0/` folder under `data/physionet.org/files/ecg-arrhythmia`, and run the script `downsample.py` from the same folder

LTDB: Download the dataset from [This Link](https://physionet.org/content/ltdb/1.0.0/) and place 1.0.0 folder under `/data/physionet.org/files/ltdb/`, and run the script `downsample.py` from the same folder 

TDBRAIN: The dataset is available under demand through [This Link](https://www.brainclinics.com/resources), and put the folders `sub-xxxxxxxx` under `/data/TDBRAIN-dataset/`, and run `fix_vhdr.py` under the root of the project.


Tunning the parameters:

python3 src/training/tuning.py -n_ts 140 -n 1000 -m 10 -r 2025 -c ecg

Post-processing:

test/post_processing.py

