import os
import re
import shutil

base = "/home/haoying/Documents/MPGAN/src/results/ipopt/ltdb_128"

for name in os.listdir(base):
    m = re.match(r"remarkable_(\d+)_", name)
    if m:
        old_path = os.path.join(base, name)
        new_name = f"ecg_{m.group(1)}"
        new_path = os.path.join(base, new_name)
        print(f"{name} -> {new_name}")
        os.rename(old_path, new_path)

print("Done.")
