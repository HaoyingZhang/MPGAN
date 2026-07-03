import os

data_dir = "data/TDBRAIN-dataset"
folders = sorted(f for f in os.listdir(data_dir) if f.startswith("sub-") and os.path.isdir(os.path.join(data_dir, f)))

out_path = os.path.join(data_dir, "participants.tsv")
with open(out_path, "w") as f:
    f.write("participant_id\n")
    for folder in folders:
        f.write(folder + "\n")

print(f"Written {len(folders)} participants to {out_path}")
