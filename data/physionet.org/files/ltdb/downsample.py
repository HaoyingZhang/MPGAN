import os
import wfdb
import numpy as np
from math import gcd
from scipy.signal import resample_poly

SRC_DIR = "1.0.0"
DST_DIR = "records100"
ORIG_FS = 128
TARGET_FS = 100


def downsample_records():
    os.makedirs(DST_DIR, exist_ok=True)

    records = [f[:-4] for f in os.listdir(SRC_DIR) if f.endswith(".hea")]

    g = gcd(ORIG_FS, TARGET_FS)
    up = TARGET_FS // g
    down = ORIG_FS // g

    for record in sorted(records):
        print(f"Processing {record}...")
        rec = wfdb.rdrecord(os.path.join(SRC_DIR, record))
        signal = rec.p_signal[:, 0]  # univariate: first channel

        resampled = resample_poly(signal, up, down)

        out_path = os.path.join(DST_DIR, f"{record}.npy")
        np.save(out_path, resampled)

    print("Done.")


if __name__ == "__main__":
    downsample_records()
