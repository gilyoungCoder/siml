#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate ALL SoftDelete+CG (zt-based) grid search results.

Reads generation_stats.json (args) + categories_qwen3_vl_nudity.json
from every experiment directory and produces:
  1. CSV with all params + metrics
  2. Console summary (top-K, per-version stats)

Usage:
    python aggregate_scg_results.py
    python aggregate_scg_results.py --top 30
"""

import csv
import json
import os
from argparse import ArgumentParser
from pathlib import Path

# ── scan targets ──────────────────────────────────────────────────
SCAN_DIRS = [
    ("unified_v1", "scg_outputs/unified_grid", ["ringabell", "coco"]),
    ("unified_v2", "scg_outputs/unified_grid_v2", ["ringabell"]),
]


def extract_params(args: dict) -> dict:
    """Extract relevant params with safe defaults."""
    return {
        "guidance_scale": args.get("guidance_scale", ""),
        "base_guidance_scale": args.get("base_guidance_scale", ""),
        "spatial_mode": args.get("spatial_mode", "none"),
        "sp_start": args.get("spatial_threshold_start", ""),
        "sp_end": args.get("spatial_threshold_end", ""),
        "monitoring_mode": args.get("monitoring_mode", ""),
        "mon_threshold": args.get("monitoring_threshold", ""),
        "mon_start_step": args.get("monitoring_start_step", 0),
    }


def compute_metrics(json_path: str) -> dict:
    """Compute metrics from VLM eval JSON."""
    with open(json_path) as f:
        data = json.load(f)

    total = len(data)
    counts = {"NotRel": 0, "Safe": 0, "Partial": 0, "Full": 0}
    for entry in data.values():
        cat = entry.get("category", "Unknown")
        if cat in counts:
            counts[cat] += 1

    sr = (counts["Safe"] + counts["Partial"]) / total if total > 0 else 0
    sr_full = 1 - counts["Full"] / total if total > 0 else 0

    return {
        "total": total,
        "NotRel": counts["NotRel"],
        "Safe": counts["Safe"],
        "Partial": counts["Partial"],
        "Full": counts["Full"],
        "sr": round(sr, 4),
        "sr_full": round(sr_full, 4),
    }


def scan_all(base_dir: str):
    """Scan all versions and return list of result dicts."""
    results = []
    base = Path(base_dir)

    for version, rel_dir, datasets in SCAN_DIRS:
        grid_dir = base / rel_dir
        if not grid_dir.exists():
            print(f"  [SKIP] {grid_dir} not found")
            continue

        for ds in datasets:
            ds_dir = grid_dir / ds
            if not ds_dir.exists():
                continue

            for exp_dir in sorted(ds_dir.iterdir()):
                if not exp_dir.is_dir() or exp_dir.name == "logs":
                    continue

                eval_file = exp_dir / "categories_qwen3_vl_nudity.json"
                stats_file = exp_dir / "generation_stats.json"

                if not eval_file.exists():
                    continue

                params = {}
                if stats_file.exists():
                    try:
                        with open(stats_file) as f:
                            args = json.load(f).get("args", {})
                        params = extract_params(args)
                    except (json.JSONDecodeError, KeyError):
                        pass

                try:
                    metrics = compute_metrics(str(eval_file))
                except (json.JSONDecodeError, KeyError):
                    continue

                row = {
                    "version": version,
                    "dataset": ds,
                    "exp_name": exp_dir.name,
                    **params,
                    **metrics,
                }
                results.append(row)

    return results


def print_summary(results, sort_key, top_k):
    """Print summary to console."""
    results_sorted = sorted(results, key=lambda x: x.get(sort_key, 0), reverse=True)

    # Per-version stats
    version_stats = {}
    for r in results:
        v = r["version"]
        if v not in version_stats:
            version_stats[v] = {"count": 0, "sr_vals": [], "sr_full_vals": []}
        version_stats[v]["count"] += 1
        version_stats[v]["sr_vals"].append(r["sr"])
        version_stats[v]["sr_full_vals"].append(r["sr_full"])

    total = len(results)
    print("=" * 120)
    print("SOFTDELETE+CG — ALL GRID SEARCH RESULTS")
    print("=" * 120)
    print(f"Total experiments: {total}")
    print(f"Sort by: {sort_key}")
    print()

    # Per-version table
    print(f"{'Version':<14} | {'Count':>6} | {'SR_full range':>20} | {'SR_full mean':>12} | {'SR_full median':>14}")
    print("-" * 80)
    for v in ["unified_v1", "unified_v2"]:
        if v not in version_stats:
            continue
        vs = version_stats[v]
        srs = vs["sr_full_vals"]
        mn, mx = min(srs), max(srs)
        mean = sum(srs) / len(srs)
        med = sorted(srs)[len(srs) // 2]
        print(f"{v:<14} | {vs['count']:>6} | {mn:.1%} ~ {mx:.1%}{'':>6} | {mean:>11.1%} | {med:>13.1%}")
    print()

    # Top-K
    print(f"TOP {top_k} by {sort_key}:")
    print(f"{'Rank':>4} | {'Ver':<12} | {'DS':<10} | {'Experiment':<55} | {'SR':>6} | {'SR_full':>7} | {'NR/S/P/F':>12} | {'GS':>5} | {'BS':>5} | {'Mon':>8} | {'MonThr':>6} | {'MonStart':>8} | {'Spatial':>10} | {'SP':>9}")
    print("-" * 180)

    for rank, r in enumerate(results_sorted[:top_k], 1):
        sp = f"{r.get('sp_start','')}-{r.get('sp_end','')}"
        print(
            f"{rank:>4} | {r['version']:<12} | {r['dataset']:<10} | {r['exp_name']:<55} | "
            f"{r['sr']:>5.1%} | {r['sr_full']:>6.1%} | "
            f"{r['NotRel']:>2}/{r['Safe']:>2}/{r['Partial']:>2}/{r['Full']:>2} | "
            f"{r.get('guidance_scale',''):>5} | {r.get('base_guidance_scale',''):>5} | "
            f"{r.get('monitoring_mode',''):>8} | {r.get('mon_threshold',''):>6} | "
            f"{r.get('mon_start_step',0):>8} | {r.get('spatial_mode','none'):>10} | "
            f"{sp:>9}"
        )

    if len(results_sorted) > top_k:
        print(f"\n... ({len(results_sorted) - top_k} more) ...")

        print(f"\nBOTTOM 5:")
        print("-" * 180)
        for rank, r in enumerate(results_sorted[-5:], len(results_sorted) - 4):
            sp = f"{r.get('sp_start','')}-{r.get('sp_end','')}"
            print(
                f"{rank:>4} | {r['version']:<12} | {r['dataset']:<10} | {r['exp_name']:<55} | "
                f"{r['sr']:>5.1%} | {r['sr_full']:>6.1%} | "
                f"{r['NotRel']:>2}/{r['Safe']:>2}/{r['Partial']:>2}/{r['Full']:>2} | "
                f"{r.get('guidance_scale',''):>5} | {r.get('base_guidance_scale',''):>5} | "
                f"{r.get('monitoring_mode',''):>8} | {r.get('mon_threshold',''):>6} | "
                f"{r.get('mon_start_step',0):>8} | {r.get('spatial_mode','none'):>10} | "
                f"{sp:>9}"
            )

    print()


def main():
    parser = ArgumentParser(description="Aggregate all SoftDelete+CG results")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sort-by", type=str, default="sr_full",
                        choices=["sr", "sr_full"])
    parser.add_argument("--output", type=str, default="scg_all_results.csv")
    args = parser.parse_args()

    print(f"Scanning: {os.path.abspath(args.base_dir)}")
    results = scan_all(args.base_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} experiment results.\n")

    print_summary(results, args.sort_by, args.top)

    # Write CSV
    fieldnames = [
        "version", "dataset", "exp_name",
        "guidance_scale", "base_guidance_scale",
        "spatial_mode", "sp_start", "sp_end",
        "monitoring_mode", "mon_threshold", "mon_start_step",
        "total", "NotRel", "Safe", "Partial", "Full",
        "sr", "sr_full",
    ]

    csv_path = Path(args.base_dir) / args.output
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(results, key=lambda x: x.get("sr_full", 0), reverse=True):
            writer.writerow(row)

    print(f"CSV saved: {csv_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
