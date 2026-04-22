import os, re, shutil

data_dir = "data/TDBRAIN-dataset"

# ── Pass 1: fix wrong subject IDs in .vhdr files (original behaviour) ──────
vhdr_fixed = 0
for root, dirs, files in os.walk(data_dir):
    for fname in files:
        if not fname.endswith(".vhdr"):
            continue
        m = re.match(r"(sub-\d+)_(.*)", fname)
        if not m:
            continue
        correct_id = m.group(1)
        path = os.path.join(root, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = re.sub(
            r"((?:DataFile|MarkerFile)=)sub[=-]\d+(_)",
            lambda mo: mo.group(1) + correct_id + mo.group(2),
            content,
        )
        if new_content != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            vhdr_fixed += 1
            print(f"Fixed subject ID: {path}")

print(f"\nPass 1 done. {vhdr_fixed} .vhdr file(s) patched.\n")

# ── Pass 2: rename ses-2 → ses-1 for subjects that have no ses-1 ────────────
# Collect candidates first so os.walk isn't confused by in-place renames.
candidates = []
for subdir in os.listdir(data_dir):
    sub_path = os.path.join(data_dir, subdir)
    if not os.path.isdir(sub_path):
        continue
    ses1 = os.path.join(sub_path, "ses-1")
    ses2 = os.path.join(sub_path, "ses-2")
    if os.path.isdir(ses2) and not os.path.isdir(ses1):
        candidates.append((sub_path, ses2, ses1))

renamed_dirs = 0
renamed_files = 0
patched_content = 0

for sub_path, ses2_dir, ses1_dir in candidates:
    eeg_dir = os.path.join(ses2_dir, "eeg")

    # ── 2a. Rename files inside eeg/ ──────────────────────────────────────
    if os.path.isdir(eeg_dir):
        for fname in sorted(os.listdir(eeg_dir)):
            new_fname = fname.replace("ses-2", "ses-1")
            if new_fname == fname:
                continue
            old_path = os.path.join(eeg_dir, fname)
            new_path = os.path.join(eeg_dir, new_fname)
            os.rename(old_path, new_path)
            renamed_files += 1

        # ── 2b. Patch ses-2 references inside text files ──────────────────
        text_exts = {".vhdr", ".vmrk", ".json", ".tsv"}
        for fname in os.listdir(eeg_dir):
            if not any(fname.endswith(ext) for ext in text_exts):
                continue
            fpath = os.path.join(eeg_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue
            new_content = content.replace("ses-2", "ses-1")
            if new_content != content:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                patched_content += 1
                print(f"  Patched content: {fpath}")

    # ── 2c. Rename the ses-2 folder itself ────────────────────────────────
    shutil.move(ses2_dir, ses1_dir)
    renamed_dirs += 1
    print(f"Renamed folder: {ses2_dir} → {ses1_dir}")

print(f"\nPass 2 done.")
print(f"  {renamed_dirs} ses-2 folder(s) renamed to ses-1")
print(f"  {renamed_files} file(s) renamed")
print(f"  {patched_content} file(s) had internal ses-2 references patched")
