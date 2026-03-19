#!/usr/bin/env python3
"""Split Ring a Bell dataset 50/50 for train/test."""

import csv
import os
import random

RAB_CSV = "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
OUTPUT_DIR = "/mnt/home/yhgil99/unlearning/rab_grid_search/data"
SEED = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Ring a Bell prompts
    prompts = []
    with open(RAB_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["sensitive prompt"].strip())

    print(f"Loaded {len(prompts)} Ring a Bell prompts")

    # Shuffle and split 50/50
    random.seed(SEED)
    indices = list(range(len(prompts)))
    random.shuffle(indices)

    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    test_idx = sorted(indices[mid:])

    train_prompts = [prompts[i] for i in train_idx]
    test_prompts = [prompts[i] for i in test_idx]

    print(f"Train: {len(train_prompts)} prompts")
    print(f"Test:  {len(test_prompts)} prompts")

    # Save as txt files
    train_path = os.path.join(OUTPUT_DIR, "ringabell_train.txt")
    test_path = os.path.join(OUTPUT_DIR, "ringabell_test.txt")
    full_path = os.path.join(OUTPUT_DIR, "ringabell_full.txt")

    with open(train_path, "w") as f:
        for p in train_prompts:
            f.write(p + "\n")

    with open(test_path, "w") as f:
        for p in test_prompts:
            f.write(p + "\n")

    with open(full_path, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    # Save indices for reference
    import json
    meta = {
        "source": RAB_CSV,
        "seed": SEED,
        "total": len(prompts),
        "train_count": len(train_prompts),
        "test_count": len(test_prompts),
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    with open(os.path.join(OUTPUT_DIR, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"  Full:  {full_path}")
    print(f"  Meta:  {os.path.join(OUTPUT_DIR, 'split_meta.json')}")


if __name__ == "__main__":
    main()
