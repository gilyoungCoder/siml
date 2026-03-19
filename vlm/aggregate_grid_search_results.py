#!/usr/bin/env python3
"""
Aggregate Grid Search Evaluation Results

Collects all VLM evaluation results from grid search folders and creates
a summary CSV with harmful rates for each hyperparameter combination.

Usage:
    python aggregate_grid_search_results.py [concept] [model]
    python aggregate_grid_search_results.py all qwen2_vl
    python aggregate_grid_search_results.py violence qwen2_vl
"""
import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
import csv

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path("/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results")

CONCEPT_FOLDERS = {
    "harassment": "harassment_9class_step24300_skip",
    "hate": "hate_9class_step20800_skip",
    "illegal": "illegal_9class_step22600_skip",
    "selfharm": "selfharm_9class_step20700_skip",
    "shocking": "shocking_9class_step23700_skip",
    "violence": "violence_9class_step15500_skip",
}

EVAL_CONCEPTS = {
    "harassment": "harassment",
    "hate": "hate",
    "illegal": "illegal",
    "selfharm": "self_harm",
    "shocking": "shocking",
    "violence": "violence",
}


def parse_folder_name(folder_name: str) -> dict:
    """Parse hyperparameter values from folder name like gs10.0_st0.2-0.4_hs1.0_bgs1.0_cos"""
    pattern = r"gs([\d.]+)_st([\d.]+)-([\d.]+)_hs([\d.]+)_bgs([\d.]+)_(cos|lin)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "guidance_scale": float(match.group(1)),
            "spatial_threshold_start": float(match.group(2)),
            "spatial_threshold_end": float(match.group(3)),
            "harmful_scale": float(match.group(4)),
            "base_guidance_scale": float(match.group(5)),
            "threshold_strategy": match.group(6),
        }
    return {}


def parse_results_file(results_path: Path) -> dict:
    """Parse results.txt file to extract counts and harmful rate."""
    if not results_path.exists():
        return None

    data = {
        "total": 0,
        "NotRelevant": 0,
        "Safe": 0,
        "Partial": 0,
        "Full": 0,
        "harmful_rate": 0.0,
    }

    with open(results_path, "r") as f:
        content = f.read()

    # Parse distribution
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("Total images:"):
            data["total"] = int(line.split(":")[1].strip())
        elif line.startswith("- NotRelevant:") or line.startswith("- NotPeople:"):
            match = re.search(r"(\d+)", line)
            if match:
                data["NotRelevant"] = int(match.group(1))
        elif line.startswith("- Safe:"):
            match = re.search(r"(\d+)", line)
            if match:
                data["Safe"] = int(match.group(1))
        elif line.startswith("- Partial:"):
            match = re.search(r"(\d+)", line)
            if match:
                data["Partial"] = int(match.group(1))
        elif line.startswith("- Full:"):
            match = re.search(r"(\d+)", line)
            if match:
                data["Full"] = int(match.group(1))
        elif "Harmful Rate" in line:
            match = re.search(r"([\d.]+)%", line)
            if match:
                data["harmful_rate"] = float(match.group(1))

    return data


def aggregate_concept(concept_key: str, model: str) -> list:
    """Aggregate results for a single concept."""
    folder_name = CONCEPT_FOLDERS.get(concept_key)
    eval_concept = EVAL_CONCEPTS.get(concept_key)

    if not folder_name:
        print(f"Unknown concept: {concept_key}")
        return []

    folder_path = BASE_DIR / folder_name
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return []

    results = []

    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        # Parse hyperparameters from folder name
        params = parse_folder_name(subfolder.name)
        if not params:
            continue

        # Look for results file
        results_file = subfolder / f"results_{model}_{eval_concept}.txt"
        if not results_file.exists():
            # Try alternative naming
            results_file = subfolder / f"results_{model}.txt"

        eval_data = parse_results_file(results_file) if results_file.exists() else None

        row = {
            "concept": concept_key,
            "folder": subfolder.name,
            **params,
        }

        if eval_data:
            row.update({
                "total": eval_data["total"],
                "not_relevant": eval_data["NotRelevant"],
                "safe": eval_data["Safe"],
                "partial": eval_data["Partial"],
                "full": eval_data["Full"],
                "harmful_rate": eval_data["harmful_rate"],
                "evaluated": True,
            })
        else:
            row.update({
                "total": 0,
                "not_relevant": 0,
                "safe": 0,
                "partial": 0,
                "full": 0,
                "harmful_rate": -1,
                "evaluated": False,
            })

        results.append(row)

    return results


def main():
    concept = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2_vl"

    print(f"Aggregating results for: {concept}")
    print(f"Model: {model}")
    print()

    all_results = []

    if concept == "all":
        for c in CONCEPT_FOLDERS.keys():
            print(f"Processing {c}...")
            results = aggregate_concept(c, model)
            all_results.extend(results)
            print(f"  Found {len(results)} folders, {sum(1 for r in results if r['evaluated'])} evaluated")
    else:
        results = aggregate_concept(concept, model)
        all_results.extend(results)

    if not all_results:
        print("No results found!")
        return

    # Save to CSV
    output_path = BASE_DIR / f"grid_search_summary_{concept}_{model}.csv"

    fieldnames = [
        "concept", "folder",
        "guidance_scale", "spatial_threshold_start", "spatial_threshold_end",
        "harmful_scale", "base_guidance_scale", "threshold_strategy",
        "total", "not_relevant", "safe", "partial", "full", "harmful_rate", "evaluated"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print()
    print(f"Saved summary to: {output_path}")

    # Print statistics
    evaluated = [r for r in all_results if r["evaluated"]]
    not_evaluated = [r for r in all_results if not r["evaluated"]]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total folders: {len(all_results)}")
    print(f"Evaluated: {len(evaluated)}")
    print(f"Not evaluated: {len(not_evaluated)}")

    if evaluated:
        harmful_rates = [r["harmful_rate"] for r in evaluated]
        print()
        print("Harmful Rate Statistics:")
        print(f"  Min: {min(harmful_rates):.2f}%")
        print(f"  Max: {max(harmful_rates):.2f}%")
        print(f"  Mean: {sum(harmful_rates)/len(harmful_rates):.2f}%")

        # Find best (lowest harmful rate) configurations
        sorted_results = sorted(evaluated, key=lambda x: x["harmful_rate"])
        print()
        print("Top 5 configurations (lowest harmful rate):")
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {r['folder']}: {r['harmful_rate']:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
