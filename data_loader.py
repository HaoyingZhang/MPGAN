import json, os
import numpy as np
import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, help="The base folder path")
    parser.add_argument("--category", type=str, help="Category of the data")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    args = parser.parse_args()
    
    dst_folder = os.path.join(args.base_folder, f"{args.category}_{args.dataset}")
    os.makedirs(dst_folder, exist_ok=True)
    base_folder = args.base_folder
    ecg_folders = sorted(
            [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and re.fullmatch(rf"{args.category}_\d+", d)],
            key=lambda d: int(d.split(f"{args.category}_")[1]
        )
    )
    print(f"Scanned {len(ecg_folders)} folders")
    for folder_name in ecg_folders:
        i = folder_name.split(f"{args.category}_")[1]
        print(i)
        with open(os.path.join(args.base_folder, folder_name, "results.json")) as f:
            res = json.load(f)
        ts_original = res["data"]
        ts_init = res["fake_data"]
        np.save(os.path.join(dst_folder, f"{args.category}_{i}.npy"), ts_original)
        np.save(os.path.join(dst_folder, f"{args.category}_{i}_init.npy"), ts_init)
        print(f"Saving {folder_name}")