#!/usr/bin/env python3
"""
Aggregate fine grid search results (Qwen VLM SR) for SAFREE+Monitoring.
Datasets: p4dn, ringabell, unlearndiff

Usage:
    python vlm/aggregate_fine_grid.py
    python vlm/aggregate_fine_grid.py --top 20
"""
import os
import json
import glob
import argparse

BASE_DIR = "/mnt/home/yhgil99/unlearning/SAFREE/results/fine_grid"
DATASETS = ["p4dn", "ringabell", "unlearndiff"]
EVAL_FILE = "categories_qwen3_vl_nudity.json"


def compute_sr(json_path):
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
    return safe / total * 100, total, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30, help="Show top N configs")
    parser.add_argument("--sort-by", default="avg", choices=["avg", "p4dn", "ringabell", "unlearndiff"])
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base directory for results")
    parser.add_argument("--complete-only", action="store_true", help="Only show configs with all datasets")
    args = parser.parse_args()

    base_dir = args.base_dir

    # Collect all configs
    all_configs = set()
    for ds in DATASETS:
        ds_dir = os.path.join(base_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for d in os.listdir(ds_dir):
            if d.startswith("mon") and os.path.isdir(os.path.join(ds_dir, d)):
                all_configs.add(d)
    configs = sorted(all_configs)

    if not configs:
        print("No results found.")
        return

    # Compute SR for each (dataset, config)
    results = {}
    for ds in DATASETS:
        for cfg in configs:
            jp = os.path.join(base_dir, ds, cfg, EVAL_FILE)
            if os.path.exists(jp):
                sr, total, counts = compute_sr(jp)
                results[(ds, cfg)] = {"sr": sr, "total": total, "counts": counts}
            else:
                results[(ds, cfg)] = None

    # Compute averages
    avg_sr = {}
    for cfg in configs:
        vals = [results[(ds, cfg)]["sr"] for ds in DATASETS if results[(ds, cfg)] is not None]
        avg_sr[cfg] = sum(vals) / len(vals) if vals else None

    # Filter: only configs with all datasets present
    if args.complete_only:
        configs = [c for c in configs if all(results[(ds, c)] is not None for ds in DATASETS)]
        if not configs:
            print("No configs with all datasets complete.")
            return

    # Sort configs
    if args.sort_by == "avg":
        configs_sorted = sorted(configs, key=lambda c: avg_sr.get(c) or -1, reverse=True)
    else:
        configs_sorted = sorted(
            configs,
            key=lambda c: results[(args.sort_by, c)]["sr"] if results[(args.sort_by, c)] else -1,
            reverse=True,
        )

    # Count completed
    done = sum(1 for ds in DATASETS for c in configs if results[(ds, c)] is not None)
    total_possible = len(configs) * len(DATASETS)
    print(f"Completed: {done}/{total_possible} ({done/total_possible*100:.0f}%)\n")

    # Print table
    print(f"{'#':<4} {'Config':<45} {'p4dn':>8} {'ringabell':>10} {'unlearndiff':>12} {'AVG':>8}")
    print("-" * 90)

    for rank, cfg in enumerate(configs_sorted[:args.top], 1):
        print(f"{rank:<4} {cfg:<45}", end="")
        for ds in DATASETS:
            r = results[(ds, cfg)]
            if r is not None:
                print(f"  {r['sr']:>7.1f}%", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        avg = avg_sr.get(cfg)
        if avg is not None:
            print(f"  {avg:>6.1f}%")
        else:
            print(f"  {'N/A':>8}")

    # Dataset averages
    print("-" * 90)
    print(f"{'':4} {'DATASET AVG':<45}", end="")
    for ds in DATASETS:
        vals = [results[(ds, c)]["sr"] for c in configs if results[(ds, c)] is not None]
        if vals:
            print(f"  {sum(vals)/len(vals):>7.1f}%", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print()

    # Save full results as JSON
    out = []
    for cfg in configs_sorted:
        entry = {"config": cfg}
        for ds in DATASETS:
            r = results[(ds, cfg)]
            entry[f"{ds}_sr"] = round(r["sr"], 2) if r else None
        entry["avg_sr"] = round(avg_sr[cfg], 2) if avg_sr[cfg] else None
        out.append(entry)

    out_path = os.path.join(base_dir, "fine_grid_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
