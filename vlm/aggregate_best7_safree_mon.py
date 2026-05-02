#!/usr/bin/env python3
"""
Aggregate Qwen VLM nudity evaluation results for best7 SAFREE+Mon configs.
Outputs a table: dataset × config with SR (Safe Rate = Safe+Partial / Total).

Usage:
    python vlm/aggregate_best7_safree_mon.py
"""
import os
import json
import glob

BASE_DIR = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs"
ALL_DATASETS = ["i2p", "mma", "p4dn", "ringabell", "unlearndiff"]

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--no-i2p-p4dn", action="store_true", help="Exclude i2p and p4dn")
_args = _parser.parse_args()

if _args.no_i2p_p4dn:
    DATASETS = [d for d in ALL_DATASETS if d not in ("i2p", "p4dn")]
else:
    DATASETS = ALL_DATASETS
EVAL_FILE = "categories_qwen3_vl_nudity.json"

def compute_sr(json_path):
    """Compute Safe Rate from categories JSON."""
    with open(json_path) as f:
        data = json.load(f)
    total = len(data)
    if total == 0:
        return 0.0, 0, {}
    counts = {}
    for v in data.values():
        cat = v["category"]
        counts[cat] = counts.get(cat, 0) + 1
    safe = counts.get("Safe", 0) + counts.get("Partial", 0)
    sr = safe / total * 100
    return sr, total, counts

# Collect all configs
all_configs = set()
for ds in DATASETS:
    pattern = f"{BASE_DIR}/final_{ds}/safree_mon/mon*/"
    for d in glob.glob(pattern):
        all_configs.add(os.path.basename(d.rstrip("/")))
configs = sorted(all_configs)

# Build results table
results = {}  # (dataset, config) -> sr
details = {}  # (dataset, config) -> (total, counts)

for ds in DATASETS:
    for cfg in configs:
        json_path = f"{BASE_DIR}/final_{ds}/safree_mon/{cfg}/{EVAL_FILE}"
        if os.path.exists(json_path):
            sr, total, counts = compute_sr(json_path)
            results[(ds, cfg)] = sr
            details[(ds, cfg)] = (total, counts)
        else:
            results[(ds, cfg)] = None

# Print table
print(f"{'Config':<45}", end="")
for ds in DATASETS:
    print(f"  {ds:>10}", end="")
print(f"  {'AVG':>8}")
print("-" * (45 + 12 * len(DATASETS) + 10))

for cfg in configs:
    print(f"{cfg:<45}", end="")
    vals = []
    for ds in DATASETS:
        sr = results[(ds, cfg)]
        if sr is not None:
            print(f"  {sr:>9.1f}%", end="")
            vals.append(sr)
        else:
            print(f"  {'N/A':>10}", end="")
    if vals:
        avg = sum(vals) / len(vals)
        print(f"  {avg:>7.1f}%")
    else:
        print(f"  {'N/A':>8}")

# Print average per dataset
print("-" * (45 + 12 * len(DATASETS) + 10))
print(f"{'AVG':<45}", end="")
for ds in DATASETS:
    vals = [results[(ds, c)] for c in configs if results[(ds, c)] is not None]
    if vals:
        print(f"  {sum(vals)/len(vals):>9.1f}%", end="")
    else:
        print(f"  {'N/A':>10}", end="")
print()

# Also print detailed breakdown per config
print("\n\n=== Detailed Breakdown ===\n")
for cfg in configs:
    print(f"--- {cfg} ---")
    for ds in DATASETS:
        key = (ds, cfg)
        if key in details:
            total, counts = details[key]
            cats = ["NotRel", "Safe", "Partial", "Full"]
            parts = [f"{c}:{counts.get(c,0)}" for c in cats]
            sr = results[key]
            print(f"  {ds:<12} SR={sr:5.1f}%  (n={total})  {' | '.join(parts)}")
        else:
            print(f"  {ds:<12} N/A")
    print()
