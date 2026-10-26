import pandas as pd
import wfdb
import os
import numpy as np
import mne
import math
from scipy.signal import resample_poly

SOURCE_HZ = {"ptbxl": 100, "arrhythmia": 100, "ltdb": 100, "tdbrain": 500, "ptb_diagnostic": 1000, "eegmmidb": 160}

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
    print(f"Resampling to {dest_hz} Hz")
    for file in files:
        raw = mne.io.read_raw_brainvision(file, preload=True, verbose=False)
        signal = raw.get_data(picks=0)[0].astype(np.float64)
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data

def _ptb_diagnostic_records():
    records_path = os.path.join("data", "ptb-diagnostic-ecg-database-1.0.0", "RECORDS")
    with open(records_path, "r") as f:
        return f.read().splitlines()


def ptb_diagnostic_records_per_patient():
    """Map patient number (int, from the 'patientXXX' folder name) to its number of records."""
    counts = {}
    for record in _ptb_diagnostic_records():
        patient_id = int(record.split("/")[0].replace("patient", ""))
        counts[patient_id] = counts.get(patient_id, 0) + 1
    return counts


def _ptb_diagnostic_one_record_per_patient():
    """One record path per patient (the last listed in RECORDS), ordered by patient number."""
    last_record = {}
    for record in _ptb_diagnostic_records():
        patient_id = int(record.split("/")[0].replace("patient", ""))
        last_record.setdefault(patient_id, record)
    return [last_record[patient_id] for patient_id in sorted(last_record)]


def ptb_diagnostic_loader(id_list, dest_hz=100, one_per_patient=False):
    file_root = os.path.join("data", "ptb-diagnostic-ecg-database-1.0.0")
    list_patient = _ptb_diagnostic_one_record_per_patient() if one_per_patient else _ptb_diagnostic_records()
    data = []
    source_hz = SOURCE_HZ["ptb_diagnostic"]
    for idx in id_list:
        record = wfdb.rdrecord(os.path.join(file_root, list_patient[idx]))
        signal = record.p_signal[:, 0].astype(np.float64)
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data


EEGMMIDB_ROOT = os.path.join("data", "eeg-motor-movementimagery-dataset-1.0.0", "files")


def eegmmidb_subjects():
    """Subject folders (S001..S109) of the EEG Motor Movement/Imagery dataset, sorted."""
    return sorted(d for d in os.listdir(EEGMMIDB_ROOT)
                  if d.startswith("S") and os.path.isdir(os.path.join(EEGMMIDB_ROOT, d)))


def eegmmidb_loader(id_list, run=2, channel=0, dest_hz=SOURCE_HZ["eegmmidb"]):
    """One signal per subject: `channel` of run `run` (R01 = baseline eyes open, R02 = baseline eyes closed)."""
    list_patient = eegmmidb_subjects()
    data = []
    for idx in id_list:
        subject = list_patient[idx]
        file = os.path.join(EEGMMIDB_ROOT, subject, f"{subject}R{run:02d}.edf")
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
        signal = raw.get_data(picks=channel)[0].astype(np.float64)
        source_hz = int(raw.info["sfreq"])
        if dest_hz != source_hz:
            g = math.gcd(dest_hz, source_hz)
            signal = resample_poly(signal, dest_hz // g, source_hz // g).astype(np.float64)
        data.append(signal)
    return data