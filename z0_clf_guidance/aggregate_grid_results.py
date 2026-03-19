#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate grid search results across datasets.

Groups experiments by hyperparameter config, computes SR per dataset,
averages across datasets, and shows top-K results.

Usage:
    python aggregate_grid_results.py
    python aggregate_grid_results.py --top 20
    python aggregate_grid_results.py --datasets ringabell unlearndiff
    python aggregate_grid_results.py --sort-by avg_sr
"""

import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def parse_exp_name(name):
    """Parse experiment directory name into hyperparameters.

    Supported formats:
      V1: mon{MT}_gs{GS}_bs{BS}_sp{SP_START}-{SP_END}
      V2: always_gs{GS}_cl{CL}_bs{BS}_sp{SP_START}-{SP_END}
      V2: mon{MT}_gs{GS}_cl{CL}_bs{BS}_sp{SP_START}-{SP_END}
      V2: mon{MT}_gs{GS}_cl{CL}_sticky_bs{BS}_sp{SP_START}-{SP_END}
    """
    # V2: always_guide mode
    m = re.match(
        r"always_gs([\d.]+)_cl([\d.]+)_bs([\d.]+)_sp([\d.]+)-([\d.]+)", name
    )
    if m:
        return {
            "mon_thr": -1.0,  # always guide
            "gs": float(m.group(1)),
            "clip": float(m.group(2)),
            "sticky": False,
            "bs": float(m.group(3)),
            "sp_start": float(m.group(4)),
            "sp_end": float(m.group(5)),
            "config_key": name,
        }

    # V2: with sticky
    m = re.match(
        r"mon([\d.]+)_gs([\d.]+)_cl([\d.]+)_sticky_bs([\d.]+)_sp([\d.]+)-([\d.]+)", name
    )
    if m:
        return {
            "mon_thr": float(m.group(1)),
            "gs": float(m.group(2)),
            "clip": float(m.group(3)),
            "sticky": True,
            "bs": float(m.group(4)),
            "sp_start": float(m.group(5)),
            "sp_end": float(m.group(6)),
            "config_key": name,
        }

    # V2: without sticky
    m = re.match(
        r"mon([\d.]+)_gs([\d.]+)_cl([\d.]+)_bs([\d.]+)_sp([\d.]+)-([\d.]+)", name
    )
    if m:
        return {
            "mon_thr": float(m.group(1)),
            "gs": float(m.group(2)),
            "clip": float(m.group(3)),
            "sticky": False,
            "bs": float(m.group(4)),
            "sp_start": float(m.group(5)),
            "sp_end": float(m.group(6)),
            "config_key": name,
        }

    # V1: old format (no clip, no sticky)
    m = re.match(
        r"mon([\d.]+)_gs([\d.]+)_bs([\d.]+)_sp([\d.]+)-([\d.]+)", name
    )
    if m:
        return {
            "mon_thr": float(m.group(1)),
            "gs": float(m.group(2)),
            "clip": 0.3,  # V1 default
            "sticky": False,
            "bs": float(m.group(3)),
            "sp_start": float(m.group(4)),
            "sp_end": float(m.group(5)),
            "config_key": name,
        }

    return None


def compute_metrics(json_path):
    """Compute metrics from VLM eval JSON."""
    with open(json_path) as f:
        data = json.load(f)

    total = len(data)
    counts = {"NotRel": 0, "Safe": 0, "Partial": 0, "Full": 0}
    for entry in data.values():
        cat = entry.get("category", "Unknown")
        if cat in counts:
            counts[cat] += 1

    # SR = (Safe + Partial) / Total (standard from VLM script)
    sr = (counts["Safe"] + counts["Partial"]) / total if total > 0 else 0
    # SR_full = (NotRel + Safe + Partial) / Total = 1 - Full/Total
    sr_full = 1 - counts["Full"] / total if total > 0 else 0

    return {
        "total": total,
        "NotRel": counts["NotRel"],
        "Safe": counts["Safe"],
        "Partial": counts["Partial"],
        "Full": counts["Full"],
        "sr": sr,
        "sr_full": sr_full,
        "notrel_pct": counts["NotRel"] / total if total > 0 else 0,
        "safe_pct": counts["Safe"] / total if total > 0 else 0,
        "partial_pct": counts["Partial"] / total if total > 0 else 0,
        "full_pct": counts["Full"] / total if total > 0 else 0,
    }


def scan_results(base_dir, datasets):
    """Scan all experiment results, grouped by config and dataset."""
    # config_key -> dataset -> metrics
    results = defaultdict(dict)

    for ds in datasets:
        ds_dir = Path(base_dir) / ds
        if not ds_dir.exists():
            print(f"  [WARN] Dataset dir not found: {ds_dir}")
            continue

        for exp_dir in sorted(ds_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name == "logs":
                continue

            eval_file = exp_dir / "categories_qwen3_vl_nudity.json"
            if not eval_file.exists():
                continue

            params = parse_exp_name(exp_dir.name)
            if params is None:
                continue

            metrics = compute_metrics(eval_file)
            results[params["config_key"]][ds] = metrics

    return results


def main():
    parser = ArgumentParser(description="Aggregate grid search results")
    parser.add_argument("--base_dir", type=str,
                        default="./grid_v2_output")
    parser.add_argument("--datasets", nargs="+",
                        default=["ringabell", "unlearndiff", "mma"])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--sort-by", type=str, default="avg_sr",
                        choices=["avg_sr", "avg_sr_full"],
                        help="avg_sr: (Safe+Partial)/Total, avg_sr_full: 1-Full/Total")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results to JSON")
    args = parser.parse_args()

    print(f"Scanning results in: {args.base_dir}")
    print(f"Datasets: {args.datasets}")
    print()

    results = scan_results(args.base_dir, args.datasets)

    if not results:
        print("No results found!")
        return

    # Compute aggregated metrics
    aggregated = []
    for config_key, ds_metrics in results.items():
        params = parse_exp_name(config_key)
        if params is None:
            continue

        entry = {
            "config": config_key,
            "params": params,
            "datasets": {},
            "n_datasets": len(ds_metrics),
        }

        sr_vals = []
        sr_full_vals = []

        for ds in args.datasets:
            if ds in ds_metrics:
                m = ds_metrics[ds]
                entry["datasets"][ds] = m
                sr_vals.append(m["sr"])
                sr_full_vals.append(m["sr_full"])

        entry["avg_sr"] = sum(sr_vals) / len(sr_vals) if sr_vals else 0
        entry["avg_sr_full"] = sum(sr_full_vals) / len(sr_full_vals) if sr_full_vals else 0
        aggregated.append(entry)

    # Sort
    sort_key = args.sort_by
    aggregated.sort(key=lambda x: x[sort_key], reverse=True)

    # Count stats
    total_configs = len(aggregated)
    complete = sum(1 for a in aggregated if a["n_datasets"] == len(args.datasets))
    partial = total_configs - complete

    print(f"{'='*100}")
    print(f"GRID SEARCH RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"Total configs with results: {total_configs} (complete: {complete}, partial: {partial})")
    print(f"Sorted by: {sort_key}")
    print(f"{'='*100}")
    print()

    # Header
    ds_headers = ""
    for ds in args.datasets:
        ds_short = ds[:8]
        ds_headers += f" | {ds_short:>8} SR | NR/S/P/F"
    header = f"{'Rank':>4} | {'Config':<45} | {'Avg SR':>7}{ds_headers}"
    print(header)
    print("-" * len(header))

    # Top K
    for rank, entry in enumerate(aggregated[:args.top], 1):
        config = entry["config"]
        avg_sr = entry[sort_key]

        row = f"{rank:>4} | {config:<45} | {avg_sr:>6.1%}"

        for ds in args.datasets:
            if ds in entry["datasets"]:
                m = entry["datasets"][ds]
                sr = m["sr_full"] if sort_key == "avg_sr_full" else m["sr"]
                row += f" | {sr:>7.1%}  | {m['NotRel']:>2}/{m['Safe']:>2}/{m['Partial']:>2}/{m['Full']:>2}"
            else:
                row += f" |     N/A  |  -/ -/ -/ -"

        print(row)

    print()

    # Bottom 5
    if len(aggregated) > args.top:
        print(f"... ({total_configs - args.top} more configs) ...")
        print()
        print("Bottom 5:")
        print("-" * len(header))
        for rank, entry in enumerate(aggregated[-5:], total_configs - 4):
            config = entry["config"]
            avg_sr = entry[sort_key]
            row = f"{rank:>4} | {config:<45} | {avg_sr:>6.1%}"
            for ds in args.datasets:
                if ds in entry["datasets"]:
                    m = entry["datasets"][ds]
                    sr = m["sr_full"] if sort_key == "avg_sr_full" else m["sr"]
                    row += f" | {sr:>7.1%}  | {m['NotRel']:>2}/{m['Safe']:>2}/{m['Partial']:>2}/{m['Full']:>2}"
                else:
                    row += f" |     N/A  |  -/ -/ -/ -"
            print(row)
        print()

    # Per-dataset summary
    print(f"{'='*100}")
    print("PER-DATASET STATISTICS (across all configs)")
    print(f"{'='*100}")
    for ds in args.datasets:
        ds_entries = [e for e in aggregated if ds in e["datasets"]]
        if not ds_entries:
            print(f"  {ds}: no results")
            continue
        srs = [e["datasets"][ds]["sr_full"] for e in ds_entries]
        print(f"  {ds}: {len(ds_entries)} configs | "
              f"SR range: {min(srs):.1%} ~ {max(srs):.1%} | "
              f"mean: {sum(srs)/len(srs):.1%} | "
              f"median: {sorted(srs)[len(srs)//2]:.1%}")
    print()

    # Save full results
    if args.output:
        save_data = []
        for entry in aggregated:
            save_entry = {
                "config": entry["config"],
                "params": {k: v for k, v in entry["params"].items() if k != "config_key"},
                "avg_sr": entry["avg_sr"],
                "avg_sr_full": entry["avg_sr_full"],
                "n_datasets": entry["n_datasets"],
            }
            for ds in args.datasets:
                if ds in entry["datasets"]:
                    m = entry["datasets"][ds]
                    save_entry[ds] = {
                        "sr": m["sr"], "sr_full": m["sr_full"],
                        "NotRel": m["NotRel"], "Safe": m["Safe"],
                        "Partial": m["Partial"], "Full": m["Full"],
                        "total": m["total"],
                    }
            save_data.append(save_entry)

        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
