import pandas as pd
import wfdb
import os
import numpy as np
import mne
import math
from scipy.signal import resample_poly

SOURCE_HZ = {"ptbxl": 100, "arrhythmia": 100, "ltdb": 100, "tdbrain": 500}

def ptbxl_loader(id_list, dest_hz=SOURCE_HZ["ptbxl"]):
    list_patient = pd.read_csv("data/physionet.org/files/ptbxl_database.csv")["filename_lr"]
    files = list_patient[id_list]
    data = []
    source_hz = SOURCE_HZ["ptbxl"]
    for file in files:
        record = wfdb.rdrecord(os.path.join("data/physionet.org/files/", file))
        signal = record.p_signal[:, 0].astype(np.float64)
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data


def arrhythmia_loader(id_list, dest_hz=SOURCE_HZ["arrhythmia"]):
    records_path = "data/physionet.org/files/ecg-arrhythmia/records100/RECORDS"
    with open(records_path, "r") as f:
        list_patient = f.read().splitlines()
    data = []
    source_hz = SOURCE_HZ["arrhythmia"]
    for idx in id_list:
        file = os.path.join(
            "data/physionet.org/files/ecg-arrhythmia/records100/",
            list_patient[idx] + ".npy",
        )
        signal = np.load(file)
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data


def ltdb_loader(id_list, dest_hz=SOURCE_HZ["ltdb"]):
    list_patient = ["14046", "14134", "14149", "14157", "14172", "14184", "15814"]
    data = []
    source_hz = SOURCE_HZ["ltdb"]
    for idx in id_list:
        file = os.path.join(
            "data/physionet.org/files/ltdb/records100/",
            list_patient[idx] + ".npy",
        )
        signal = np.load(file)
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data

def tdbrain_loader(id_list, session="EC", dest_hz=SOURCE_HZ["tdbrain"]):
    file_root = os.path.join("data", "TDBRAIN-dataset")
    list_patient = pd.read_csv(os.path.join(file_root, "participants.tsv"), sep="\t")["participant_id"].tolist()
    list_patient_to_load = [list_patient[i] for i in id_list]
    files = [os.path.join(file_root, file, "ses-1", "eeg", f"{file}_ses-1_task-rest{session}_eeg.vhdr") for file in list_patient_to_load]
    data = []
    source_hz = SOURCE_HZ["tdbrain"]
    for file in files:
        raw = mne.io.read_raw_brainvision(file, preload=True, verbose=False)
        signal = raw.get_data(picks=0)[0].astype(np.float64)
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data