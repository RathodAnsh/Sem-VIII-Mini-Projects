"""
organize_dataset.py
───────────────────
Run this once after downloading the MRL Eye Dataset from Kaggle.

Usage:
    python organize_dataset.py --open_dir /path/to/open_eyes \
                                --closed_dir /path/to/closed_eyes

Dataset link:
    https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset
"""

import os
import shutil
import random
import argparse
from tqdm import tqdm


def organize(open_dir: str, closed_dir: str, train_ratio: float = 0.85):
    splits   = ["train", "val"]
    classes  = ["open", "closed"]
    sources  = {"open": open_dir, "closed": closed_dir}

    for split in splits:
        for cls in classes:
            os.makedirs(f"dataset/{split}/{cls}", exist_ok=True)

    for cls, src in sources.items():
        if not os.path.exists(src):
            print(f"[WARN] Directory not found: {src}")
            continue
        files = [f for f in os.listdir(src)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)

        print(f"Copying {cls} eyes ({len(files)} images)...")
        for i, fname in enumerate(tqdm(files)):
            split = "train" if i < split_idx else "val"
            shutil.copy(os.path.join(src, fname),
                        f"dataset/{split}/{cls}/{fname}")

    # Print summary
    for split in splits:
        for cls in classes:
            n = len(os.listdir(f"dataset/{split}/{cls}"))
            print(f"  dataset/{split}/{cls}: {n} images")

    print("\nDataset organised successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_dir",   required=True, help="Path to open eye images")
    parser.add_argument("--closed_dir", required=True, help="Path to closed eye images")
    parser.add_argument("--ratio",      type=float, default=0.85, help="Train split ratio")
    args = parser.parse_args()
    organize(args.open_dir, args.closed_dir, args.ratio)
