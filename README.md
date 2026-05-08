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

## 2. Testing DeepMP

Test on TDBRAIN dataset (disjoint from the training set):
```./exe_test_eeg.sh```

Test on PTBXL dataset (disjoint from the training set):
```./exe_test_ptbxl.sh```

Test on Arrhythmia dataset:
```./exe_test_arrhythmia.sh```

Test on LTDB dataset:
```./exe_test_ltdb.sh```

Note and eventually rename the path output in the command, which will be used after

## 3. Using solver

To refine the results with the ipopt solver, please follow these steps

### 3.1. Prepare the data for the solver
We will use the data output by the model as the start point of the Constraint Satisfaction Problem of Matrix Profile inversion.

Launch the following command by replacing the items in brackets by your value. 

```bash
python3 data_loader.py --base_folder [path] --category [category] --dataset [dataset]
```
- [path] : The path output (eventually renamed) at the output of step "Testing"
- [category] : Data type, "ecg" for electrodiagram or "eeg" for electroencephalogram
- [dataset] : The dataset name, "ptbxl", "arrhythmia", "ltdb"

The output should be a folder named "[category]_[dataset]" under the same path as the value [path]

### 3.2. Clone the project from : 

```https://gitlab.inria.fr/petscraft-public/attacks-by-reconstruction.git```

Make sure that you clone from the latest commit to integrate the updates

### 3.3. Set the environment with the instructions

### 3.4. Copy paste the data folder 

Copy paste the folder output by the step 3.1 under ```attacks-by-reconstruction/data/```

we name the folder in the following step as [data_deepmp]

### 3.5. Modify the data path

In the file ```attacks-by-reconstruction/src/rmpi/exhaustive_rmpi.py```, modify 

- At line 150, the path "data/ecg/ecg_*.npy" becomes "data/[data_deepmp]/[category]_*.npy"

- At line 409, the path "data/ecg/ecg_{metadata['i_[ecg]']}\_init.npy" becomes "data/[data_deepmp]/[category]\_{metadata['i_[category]']}_init.npy"

### 3.6. Launch the command
```bash
python3 src/rmpi/exhaustive_rmpi.py -n 500 -m 100 -l 1200 -gz -r 2025 -a [nb_patient] -pop 6 -parallel 5 -c ecg -siv
```
where [nb_patient] = 200 for "ptbxl" and "tdbrain"
[nb_patient] = 210 for "ltdb"
[nb_patient] = 240 for "arrhythmia"

### 3.7. Integrate the solutions to the evaluation project

Copy the folder containing only the results (the ```/stored_rmpi``` folder under the output folder) and paste to ```src/results/ipopt/``` and rename ```stored_rmpi``` to [dataset]

Launch ```python3 rename_folders.py --base_folder src/results/ipopt/[dataset] --category [category]```

## 4. Evaluation of reconstruction

To reproduce the results in Table 2, run :

```bash
python3 eval.py --category [category] --dataset [dataset] [--ipopt]
```

by replacing [category] by "ecg" or "eeg" and [dataset] by "ptbxl", "arrhythmia", "ltdb" or "tdbrain", and with or without [--ipopt]

## 5. Evaluation of re-identification

To reproduce the results in Table 3, run : 

```bash
python3 reidentify.py [--verbose] --category [category] --dataset [dataset] [--ipopt] --feature
```

Use "arrhythmia" and "ptbxl" as [dataset] for ecg data and "tdbrain" as [dataset] for eeg

## 6. Robustness

```bash
python3 src/plot/plot_paper.py --exp_name robustness
```


