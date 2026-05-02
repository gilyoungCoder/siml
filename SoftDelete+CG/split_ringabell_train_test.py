#!/usr/bin/env python3
"""
Split RingaBell 'Full' nudity prompts into train/test sets.

- Reads vanilla SD 1.4 VLM eval (categories_qwen3_vl_nudity.json)
- Identifies 'Full' prompts
- Randomly splits them 50/50 (seed=42)
- Train CSV: 30 Full prompts (for generating classifier training images)
- Test CSV: remaining 50 RingaBell prompts (29 Full + 21 non-Full)
"""

import json
import csv
import random
from pathlib import Path

# === Paths ===
CSV_PATH = "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CATEGORIES_PATH = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/final_ringabell/sd_baseline/categories_qwen3_vl_nudity.json"
OUTPUT_DIR = Path("/mnt/home/yhgil99/unlearning/SoftDelete+CG/ringabell_split")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load CSV ===
prompts = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append(row["sensitive prompt"])

print(f"Total RingaBell prompts: {len(prompts)}")

# === Load categories ===
with open(CATEGORIES_PATH, "r") as f:
    categories = json.load(f)

# Map index -> category
idx_to_cat = {}
for fname, info in categories.items():
    idx = int(fname.replace(".png", ""))
    idx_to_cat[idx] = info["category"]

# === Identify Full prompts ===
full_indices = sorted([idx for idx, cat in idx_to_cat.items() if cat == "Full"])
non_full_indices = sorted([idx for idx, cat in idx_to_cat.items() if cat != "Full"])

print(f"Full prompts: {len(full_indices)}")
print(f"Non-Full prompts: {len(non_full_indices)} (Partial: {sum(1 for i in non_full_indices if idx_to_cat[i]=='Partial')}, Safe: {sum(1 for i in non_full_indices if idx_to_cat[i]=='Safe')}, NotRel: {sum(1 for i in non_full_indices if idx_to_cat[i]=='NotRel')})")

# === Split Full prompts 50/50 ===
random.seed(42)
shuffled_full = full_indices.copy()
random.shuffle(shuffled_full)

n_train = len(shuffled_full) // 2  # 29
train_indices = sorted(shuffled_full[:n_train])
test_full_indices = sorted(shuffled_full[n_train:])

print(f"\nTrain Full prompts: {n_train}")
print(f"Test Full prompts: {len(test_full_indices)}")

# === Test set = remaining Full + all non-Full ===
test_indices = sorted(test_full_indices + non_full_indices)
print(f"Test total prompts: {len(test_indices)}")

# === Write train CSV (sensitive prompt only, for SD 1.4 generation) ===
train_csv = OUTPUT_DIR / "ringabell_train_full.csv"
with open(train_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sensitive prompt"])
    for idx in train_indices:
        writer.writerow([prompts[idx]])

print(f"\nWrote train CSV: {train_csv} ({n_train} prompts)")

# === Write test CSV (same format as original, sensitive + normal) ===
# Re-read original with both columns
all_rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_rows.append(row)

test_csv = OUTPUT_DIR / "ringabell_test.csv"
with open(test_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sensitive prompt", "normal prompt"])
    writer.writeheader()
    for idx in test_indices:
        writer.writerow(all_rows[idx])

print(f"Wrote test CSV: {test_csv} ({len(test_indices)} prompts)")

# === Write metadata ===
meta = {
    "total_ringabell": len(prompts),
    "total_full": len(full_indices),
    "train_count": n_train,
    "train_indices": train_indices,
    "test_count": len(test_indices),
    "test_indices": test_indices,
    "test_full_count": len(test_full_indices),
    "test_nonfull_count": len(non_full_indices),
    "split_seed": 42,
    "source_categories": str(CATEGORIES_PATH),
}
with open(OUTPUT_DIR / "split_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote metadata: {OUTPUT_DIR / 'split_meta.json'}")

# === Summary ===
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Train: {n_train} Full prompts → generate 1500 images (50/prompt)")
print(f"Test:  {len(test_indices)} prompts ({len(test_full_indices)} Full + {len(non_full_indices)} non-Full)")
print(f"Train indices: {train_indices}")
print(f"Test Full indices: {test_full_indices}")
