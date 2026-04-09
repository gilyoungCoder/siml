#!/usr/bin/env python3
"""
Aggregate multi-concept experiment results into a summary table.

Reads Qwen VLM evaluation results from:
  CAS_SpatialCFG/outputs/multi_concept/{concept}/{method}/

Output: method × concept SR table + CSV export.

Usage:
    python scripts/aggregate_multi_concept_results.py
    python scripts/aggregate_multi_concept_results.py --base-dir CAS_SpatialCFG/outputs/multi_concept
    python scripts/aggregate_multi_concept_results.py --csv results_summary.csv
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "vlm"))

CONCEPTS = ["sexual", "violence", "harassment", "hate", "shocking", "illegal_activity", "self-harm"]

# Map concept names to what Qwen VLM eval uses in filenames
CONCEPT_EVAL_NAMES = {
    "sexual": "nudity",
    "violence": "violence",
    "harassment": "harassment",
    "hate": "hate",
    "shocking": "shocking",
    "illegal_activity": "illegal",
    "self-harm": "self_harm",
}


def find_qwen_result(method_dir, concept):
    """Find Qwen VLM result files in a method output directory."""
    eval_name = CONCEPT_EVAL_NAMES.get(concept, concept)

    # Try multiple naming conventions (qwen3_vl first — canonical)
    candidates = [
        f"categories_qwen3_vl_{eval_name}.json",
        f"categories_qwen_{eval_name}.json",
        f"categories_qwen3_vl_nudity.json" if concept == "sexual" else None,
        f"categories_qwen_nudity.json" if concept == "sexual" else None,
    ]

    for fname in candidates:
        if fname is None:
            continue
        fpath = os.path.join(method_dir, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def find_results_txt(method_dir, concept):
    """Find results text file."""
    eval_name = CONCEPT_EVAL_NAMES.get(concept, concept)
    candidates = [
        f"results_qwen3_vl_{eval_name}.txt",
        f"results_qwen_{eval_name}.txt",
        f"results_qwen3_vl_nudity.txt" if concept == "sexual" else None,
        f"results_qwen_nudity.txt" if concept == "sexual" else None,
    ]
    for fname in candidates:
        if fname is None:
            continue
        fpath = os.path.join(method_dir, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def parse_categories_json(json_path):
    """Parse categories JSON and compute SR = (Safe + Partial + NotRel) / Total."""
    with open(json_path) as f:
        data = json.load(f)

    counts = {"NotRel": 0, "Safe": 0, "Partial": 0, "Full": 0}
    for img_result in data.values():
        cat = img_result if isinstance(img_result, str) else img_result.get("category", "")
        if cat in counts:
            counts[cat] += 1

    total = sum(counts.values())
    if total == 0:
        return None

    # SR = (Safe + Partial) / Total  (canonical definition from CLAUDE.md)
    sr = (counts["Safe"] + counts["Partial"]) / total * 100
    return {
        "counts": counts,
        "total": total,
        "sr": sr,
        "full_pct": counts["Full"] / total * 100,
    }


def parse_results_txt(txt_path):
    """Fallback: parse results text file for SR."""
    with open(txt_path) as f:
        content = f.read()

    for line in content.split("\n"):
        if "SR" in line or "Success Rate" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    return {"sr": float(parts[-1].strip().rstrip("%"))}
                except ValueError:
                    pass
    return None


def collect_results(base_dir):
    """Collect all results from the multi-concept output directory."""
    results = defaultdict(dict)  # results[method][concept] = {sr, counts, ...}
    methods_found = set()

    for concept in CONCEPTS:
        concept_dir = os.path.join(base_dir, concept)
        if not os.path.isdir(concept_dir):
            continue

        for method_name in sorted(os.listdir(concept_dir)):
            method_dir = os.path.join(concept_dir, method_name)
            if not os.path.isdir(method_dir):
                continue

            # Check for images
            png_count = len([f for f in os.listdir(method_dir) if f.endswith(".png")])
            if png_count == 0:
                continue

            methods_found.add(method_name)

            # Try categories JSON first
            json_path = find_qwen_result(method_dir, concept)
            if json_path:
                result = parse_categories_json(json_path)
                if result:
                    results[method_name][concept] = result
                    continue

            # Fallback to results txt
            txt_path = find_results_txt(method_dir, concept)
            if txt_path:
                result = parse_results_txt(txt_path)
                if result:
                    results[method_name][concept] = result
                    continue

            # No eval results yet, just mark as generated
            results[method_name][concept] = {"sr": None, "total": png_count, "status": "generated"}

    return results, sorted(methods_found)


def print_summary_table(results, methods):
    """Print formatted summary table."""
    # Header
    header = f"{'Method':<45}"
    for concept in CONCEPTS:
        short = concept[:8]
        header += f" | {short:>8}"
    header += f" | {'Avg':>6}"
    print("=" * len(header))
    print("Multi-Concept Experiment Results — SR (Safe + Partial) / Total %")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<45}"
        srs = []
        for concept in CONCEPTS:
            if concept in results[method]:
                r = results[method][concept]
                sr = r.get("sr")
                if sr is not None:
                    row += f" | {sr:>7.1f}%"
                    srs.append(sr)
                else:
                    row += f" | {'gen':>8}"
            else:
                row += f" | {'-':>8}"
        avg = sum(srs) / len(srs) if srs else 0
        row += f" | {avg:>5.1f}%"
        print(row)

    print("=" * len(header))

    # Also print Full% table (unsafe rate)
    print()
    header2 = f"{'Method':<45}"
    for concept in CONCEPTS:
        short = concept[:8]
        header2 += f" | {short:>8}"
    header2 += f" | {'Avg':>6}"
    print("Full (Unsafe) Rate %")
    print("-" * len(header2))
    print(header2)
    print("-" * len(header2))

    for method in methods:
        row = f"{method:<45}"
        fulls = []
        for concept in CONCEPTS:
            if concept in results[method]:
                r = results[method][concept]
                full = r.get("full_pct")
                if full is not None:
                    row += f" | {full:>7.1f}%"
                    fulls.append(full)
                else:
                    row += f" | {'-':>8}"
            else:
                row += f" | {'-':>8}"
        avg = sum(fulls) / len(fulls) if fulls else 0
        row += f" | {avg:>5.1f}%"
        print(row)

    print("-" * len(header2))


def save_csv(results, methods, csv_path):
    """Export results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method"] + [f"{c}_sr" for c in CONCEPTS] + [f"{c}_full" for c in CONCEPTS] + ["avg_sr", "avg_full"]
        writer.writerow(header)

        for method in methods:
            row = [method]
            srs = []
            fulls = []
            for concept in CONCEPTS:
                r = results[method].get(concept, {})
                sr = r.get("sr")
                full = r.get("full_pct")
                row.append(f"{sr:.1f}" if sr is not None else "")
                srs.append(sr if sr is not None else None)
            for concept in CONCEPTS:
                r = results[method].get(concept, {})
                full = r.get("full_pct")
                row.append(f"{full:.1f}" if full is not None else "")
                fulls.append(full if full is not None else None)

            valid_srs = [s for s in srs if s is not None]
            valid_fulls = [f for f in fulls if f is not None]
            row.append(f"{sum(valid_srs)/len(valid_srs):.1f}" if valid_srs else "")
            row.append(f"{sum(valid_fulls)/len(valid_fulls):.1f}" if valid_fulls else "")
            writer.writerow(row)

    print(f"\nCSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-concept results")
    parser.add_argument("--base-dir", type=str,
                       default=str(REPO_ROOT / "CAS_SpatialCFG" / "outputs" / "multi_concept"),
                       help="Base directory with concept/method subdirectories")
    parser.add_argument("--csv", type=str, default=None,
                       help="Export results to CSV file")
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Base directory not found: {args.base_dir}")
        print("Run experiments first with scripts/run_multi_concept.sh")
        sys.exit(1)

    results, methods = collect_results(args.base_dir)

    if not methods:
        print("No results found.")
        sys.exit(1)

    print_summary_table(results, methods)

    if args.csv:
        save_csv(results, methods, args.csv)


if __name__ == "__main__":
    main()
