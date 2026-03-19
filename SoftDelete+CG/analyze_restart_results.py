#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze ASCG + Restart Sampling results.

Compares Phase 1 (guided only) vs Phase 2 (restart) images
across different restart configurations.

Usage:
  python analyze_restart_results.py --results_dir scg_outputs/restart_poc
  python analyze_restart_results.py --results_dir scg_outputs/restart_grid --vlm_eval
"""

import os
import sys
import json
import glob
import csv
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def parse_config_from_dirname(dirname: str) -> dict:
    """Parse experiment config from directory name."""
    config = {
        "name": dirname,
        "restart_timestep": 0,
        "guidance_fraction": 0.0,
        "restart_count": 0,
        "is_baseline": False,
    }

    if "baseline" in dirname or "no_restart" in dirname:
        config["is_baseline"] = True
        return config

    parts = dirname.split("_")
    for part in parts:
        if part.startswith("t") and part[1:].isdigit():
            config["restart_timestep"] = int(part[1:])
        elif part.startswith("gf"):
            try:
                config["guidance_fraction"] = float(part[2:])
            except ValueError:
                pass
        elif part.startswith("rc"):
            try:
                config["restart_count"] = int(part[2:])
            except ValueError:
                pass

    return config


def load_generation_stats(stats_path: str) -> dict:
    """Load generation statistics from JSON file."""
    with open(stats_path, "r") as f:
        return json.load(f)


def load_vlm_results(results_path: str) -> dict:
    """Parse VLM evaluation results file."""
    results = {
        "total": 0,
        "safe": 0,
        "partial": 0,
        "full": 0,
        "not_rel": 0,
        "sr": 0.0,
    }

    with open(results_path, "r") as f:
        content = f.read()

    for line in content.strip().split("\n"):
        line = line.strip()
        if "Total images:" in line:
            results["total"] = int(line.split(":")[-1].strip())
        elif "- Safe:" in line:
            parts = line.split("(")
            results["safe"] = int(parts[0].split(":")[-1].strip())
        elif "- Partial:" in line:
            parts = line.split("(")
            results["partial"] = int(parts[0].split(":")[-1].strip())
        elif "- Full:" in line:
            parts = line.split("(")
            results["full"] = int(parts[0].split(":")[-1].strip())
        elif "- NotRel:" in line:
            parts = line.split("(")
            results["not_rel"] = int(parts[0].split(":")[-1].strip())
        elif "SR (Safe+Partial):" in line:
            # SR (Safe+Partial): 62/79 (78.5%)
            pct_str = line.split("(")[-1].replace("%)", "").strip()
            results["sr"] = float(pct_str) / 100.0

    return results


def collect_results(results_dir: str) -> list:
    """Collect results from all experiment directories."""
    results_dir = Path(results_dir)
    experiments = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ["logs", "smoke_test"]:
            continue

        exp = parse_config_from_dirname(exp_dir.name)
        exp["dir"] = str(exp_dir)

        # Count generated images
        images = list(exp_dir.glob("*.png"))
        exp["num_images"] = len(images)

        # Load generation stats
        stats_path = exp_dir / "generation_stats.json"
        if stats_path.exists():
            stats = load_generation_stats(str(stats_path))
            exp["total"] = stats["summary"]["total"]
            exp["safety_fallback"] = stats["summary"]["safety_fallback"]
            exp["restart_applied"] = stats["summary"]["restart_applied"]
        else:
            exp["total"] = exp["num_images"]
            exp["safety_fallback"] = 0
            exp["restart_applied"] = 0

        # Load VLM results if available
        vlm_path = exp_dir / "results_qwen3_vl_nudity.txt"
        if vlm_path.exists():
            vlm = load_vlm_results(str(vlm_path))
            exp["vlm"] = vlm
        else:
            exp["vlm"] = None

        experiments.append(exp)

    return experiments


def print_summary_table(experiments: list):
    """Print a comparison table of all experiments."""
    print("\n" + "=" * 120)
    print("RESTART SAMPLING EXPERIMENT RESULTS")
    print("=" * 120)

    # Header
    header = (
        f"{'Experiment':<35} {'RT':>4} {'GF':>5} {'RC':>3} "
        f"{'Imgs':>5} {'Fallback':>9} "
    )

    has_vlm = any(e.get("vlm") for e in experiments)
    if has_vlm:
        header += (
            f"{'Safe':>6} {'Partial':>8} {'Full':>6} {'NotRel':>7} {'SR%':>7}"
        )

    print(header)
    print("-" * 120)

    # Sort: baseline first, then by restart_timestep
    experiments.sort(
        key=lambda x: (not x["is_baseline"], x["restart_timestep"],
                        x["guidance_fraction"], x["restart_count"])
    )

    baseline_sr = None

    for exp in experiments:
        row = (
            f"{exp['name']:<35} "
            f"{exp['restart_timestep']:>4} "
            f"{exp['guidance_fraction']:>5.1f} "
            f"{exp['restart_count']:>3} "
            f"{exp['num_images']:>5} "
            f"{exp['safety_fallback']:>9} "
        )

        if has_vlm and exp.get("vlm"):
            vlm = exp["vlm"]
            sr = vlm["sr"]
            total = vlm["total"] if vlm["total"] > 0 else 1

            if exp["is_baseline"]:
                baseline_sr = sr

            delta = ""
            if baseline_sr is not None and not exp["is_baseline"]:
                d = (sr - baseline_sr) * 100
                delta = f" ({d:+.1f}pp)"

            row += (
                f"{vlm['safe']/total:>6.1%} "
                f"{vlm['partial']/total:>8.1%} "
                f"{vlm['full']/total:>6.1%} "
                f"{vlm['not_rel']/total:>7.1%} "
                f"{sr:>6.1%}{delta}"
            )
        elif has_vlm:
            row += f"{'(pending)':>40}"

        print(row)

    print("=" * 120)

    # Print key insights
    if has_vlm:
        vlm_exps = [e for e in experiments if e.get("vlm")]
        if vlm_exps:
            best = max(vlm_exps, key=lambda x: x["vlm"]["sr"])
            worst = min(vlm_exps, key=lambda x: x["vlm"]["sr"])
            print(f"\nBest SR:  {best['name']} ({best['vlm']['sr']:.1%})")
            print(f"Worst SR: {worst['name']} ({worst['vlm']['sr']:.1%})")

            if baseline_sr is not None:
                restarts = [
                    e for e in vlm_exps
                    if not e["is_baseline"] and e.get("vlm")
                ]
                if restarts:
                    best_restart = max(restarts, key=lambda x: x["vlm"]["sr"])
                    print(
                        f"\nBaseline SR: {baseline_sr:.1%}"
                        f"\nBest restart SR: {best_restart['vlm']['sr']:.1%} "
                        f"({best_restart['name']})"
                        f"\nDelta: {(best_restart['vlm']['sr'] - baseline_sr)*100:+.1f}pp"
                    )

    # Safety fallback summary
    total_fallbacks = sum(e["safety_fallback"] for e in experiments)
    if total_fallbacks > 0:
        print(f"\nSafety fallbacks: {total_fallbacks} total")
        for exp in experiments:
            if exp["safety_fallback"] > 0:
                rate = exp["safety_fallback"] / max(exp["total"], 1)
                print(
                    f"  {exp['name']}: {exp['safety_fallback']} "
                    f"({rate:.1%} fallback rate)"
                )

    print()


def export_csv(experiments: list, output_path: str):
    """Export results to CSV for further analysis."""
    fieldnames = [
        "name", "restart_timestep", "guidance_fraction", "restart_count",
        "is_baseline", "num_images", "safety_fallback",
        "vlm_safe", "vlm_partial", "vlm_full", "vlm_not_rel", "vlm_sr",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = {
                "name": exp["name"],
                "restart_timestep": exp["restart_timestep"],
                "guidance_fraction": exp["guidance_fraction"],
                "restart_count": exp["restart_count"],
                "is_baseline": exp["is_baseline"],
                "num_images": exp["num_images"],
                "safety_fallback": exp["safety_fallback"],
            }
            if exp.get("vlm"):
                total = max(exp["vlm"]["total"], 1)
                row["vlm_safe"] = exp["vlm"]["safe"] / total
                row["vlm_partial"] = exp["vlm"]["partial"] / total
                row["vlm_full"] = exp["vlm"]["full"] / total
                row["vlm_not_rel"] = exp["vlm"]["not_rel"] / total
                row["vlm_sr"] = exp["vlm"]["sr"]
            writer.writerow(row)

    print(f"Results exported to: {output_path}")


def main():
    parser = ArgumentParser(description="Analyze restart sampling results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="scg_outputs/restart_poc",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    args = parser.parse_args()

    print(f"Analyzing results in: {args.results_dir}")

    experiments = collect_results(args.results_dir)

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments")

    print_summary_table(experiments)

    # Export CSV
    csv_path = args.export_csv or os.path.join(
        args.results_dir, "results_summary.csv"
    )
    export_csv(experiments, csv_path)


if __name__ == "__main__":
    main()
