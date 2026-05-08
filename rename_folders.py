import os
import re
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, help="The base folder path")
    parser.add_argument("--category", type=str, help="Category of the data")
    args = parser.parse_args()

    base = args.base_folder

    for name in os.listdir(base):
        m = re.match(r"remarkable_(\d+)_", name)
        if m:
            old_path = os.path.join(base, name)
            new_name = f"{args.category}_{m.group(1)}"
            new_path = os.path.join(base, new_name)
            print(f"{name} -> {new_name}")
            os.rename(old_path, new_path)

    print("Done.")
