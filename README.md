# Privacy Evaluation Framework for Matrix Profile

A framework for empirically evaluating the **reconstruction risk** and **re-identification risk** of time series data de-identified via Matrix Profile-based methods.

---

Two attack pipelines are implemented:

| Attack Type | Question | Metric |
|---|---|---|
| **Reconstruction** | Can the original signal and sensitive features be recovered from the de-identified version? | RMSE, correlation, Relative Error |
| **Re-identification** | Can the source subject be identified from the de-identified signal? | Re-identification Rate (RIR) |

---

## 0. Data preparation:

PTBXL: Download from [This Link](https://physionet.org/content/ptb-xl/1.0.3/) and place the folder physionet.org under `data/`

Arrhythmia: Download the dataset from [This Link](https://physionet.org/content/mitdb/1.0.0/) and place the `1.0.0/` folder under `data/physionet.org/files/ecg-arrhythmia`, and run the script `downsample.py` from the same folder

LTDB: Download the dataset from [This Link](https://physionet.org/content/ltdb/1.0.0/) and place 1.0.0 folder under `/data/physionet.org/files/ltdb/`, and run the script `downsample.py` from the same folder 

TDBRAIN: The dataset is available under demand through [This Link](https://www.brainclinics.com/resources), and put the folders `sub-xxxxxxxx` under `/data/TDBRAIN-dataset/`, and run `fix_vhdr.py` under the root of the project.

---

## 1. DeepMP training
For ECG data:
```bash
python3 src/main_ptbxl.py -n_ts 4000000 -n 500 -m 100 -e 100000 -r 2026 -c ecg -dataset ptbxl -train_id 0 20000 -test_id 21001 21010 -p -k 0.6 -g_model deepmp -alpha 0.5 -pi_mp 0.0 -lr_g 0.0008 -time 86400 -znorm -mp_embedding -fill 100 
```

For EEG data:
```bash
python3 src/main_ptbxl.py -n_ts 100000 -n 500 -m 100 -e 100000 -r 2026 -c eeg -dataset tdbrain -train_id 0 1000 -test_id 1000 1274 -p -k 0.6 -g_model deepmp -alpha 0.5 -pi_mp 0.0 -lr_g 0.0008 -time 86400 -znorm -mp_embedding -fill 100 -val -n_val 274
```

Once the model is trained, rename the ECG folder by `ptbxl` and the EEG folder by `eeg`

- p.s.: Tunning the parameters:

```bash
python3 src/training/tuning.py -n_ts 140 -n 500 -m 100 -r 2026 -c ecg
```

## 2. Testing

Test on TDBRAIN dataset (disjoint from the training set):
```./exe_test_eeg.sh```

Test on PTBXL dataset (disjoint from the training set):
```./exe_test_ptbxl.sh```

Test on Arrhythmia dataset:
```./exe_test_arrhythmia.sh```

Test on LTDB dataset:
```./exe_test_ltdb.sh```

Post-processing:

test/post_processing.py

