#!/usr/bin/env python3
"""
Aggregate mon01_grid results: VLM Safety Rate + FID.
Outputs Table 1 style: Config × Dataset with SR, Safe, Partial, Full, NotRel, FID.

Usage:
    python vlm/aggregate_mon01_grid.py
"""
import os
import json
import glob

BASE_DIR = "SoftDelete+CG/scg_outputs/mon01_grid"
SAFETY_DATASETS = ["ringabell", "unlearndiff", "mma"]
EVAL_FILE = "categories_qwen3_vl_nudity.json"
FID_FILE = "eval_metrics.json"


def compute_sr(json_path):
    with open(json_path) as f:
        data = json.load(f)
    total = len(data)
    if total == 0:
        return None
    counts = {}
    for v in data.values():
        cat = v["category"]
        counts[cat] = counts.get(cat, 0) + 1
    safe = counts.get("Safe", 0) + counts.get("Partial", 0)
    sr = safe / total * 100
    return {"sr": sr, "total": total, "counts": counts}


def get_fid(config_name):
    fid_path = os.path.join(BASE_DIR, "coco", config_name, FID_FILE)
    if os.path.exists(fid_path):
        with open(fid_path) as f:
            return json.load(f).get("fid")
    return None


# Collect all config names
all_configs = set()
for ds in SAFETY_DATASETS:
    for d in glob.glob(f"{BASE_DIR}/{ds}/*/"):
        all_configs.add(os.path.basename(d.rstrip("/")))
configs = sorted(all_configs)

# Build results
results = {}  # (ds, cfg) -> sr_info
fid_results = {}  # cfg -> fid

for cfg in configs:
    fid_results[cfg] = get_fid(cfg)
    for ds in SAFETY_DATASETS:
        path = os.path.join(BASE_DIR, ds, cfg, EVAL_FILE)
        if os.path.exists(path):
            results[(ds, cfg)] = compute_sr(path)
        else:
            results[(ds, cfg)] = None

# Print header
cats = ["Safe", "Partial", "Full", "NotRel"]
print(f"{'Config':<35}", end="")
for ds in SAFETY_DATASETS:
    print(f" | {ds:>12} SR", end="")
print(f" | {'AVG SR':>8} | {'FID':>7}")
print("-" * 120)

# Print rows sorted by avg SR descending
def avg_sr(cfg):
    vals = []
    for ds in SAFETY_DATASETS:
        r = results.get((ds, cfg))
        if r:
            vals.append(r["sr"])
    return sum(vals) / len(vals) if vals else -1

for cfg in sorted(configs, key=avg_sr, reverse=True):
    print(f"{cfg:<35}", end="")
    sr_vals = []
    for ds in SAFETY_DATASETS:
        r = results.get((ds, cfg))
        if r:
            print(f" | {r['sr']:>14.1f}%", end="")
            sr_vals.append(r["sr"])
        else:
            print(f" | {'N/A':>15}", end="")

    if sr_vals:
        avg = sum(sr_vals) / len(sr_vals)
        print(f" | {avg:>7.1f}%", end="")
    else:
        print(f" | {'N/A':>8}", end="")

    fid = fid_results.get(cfg)
    if fid is not None:
        print(f" | {fid:>7.2f}")
    else:
        print(f" | {'N/A':>7}")

# Detailed breakdown
print("\n\n=== Detailed Breakdown ===\n")
for cfg in sorted(configs, key=avg_sr, reverse=True):
    fid = fid_results.get(cfg)
    fid_str = f"FID={fid:.2f}" if fid else "FID=N/A"
    print(f"--- {cfg} ({fid_str}) ---")
    for ds in SAFETY_DATASETS:
        r = results.get((ds, cfg))
        if r:
            parts = [f"{c}:{r['counts'].get(c, 0)}" for c in cats]
            print(f"  {ds:<12} SR={r['sr']:5.1f}%  (n={r['total']})  {' | '.join(parts)}")
        else:
            print(f"  {ds:<12} N/A")
    print()
