#!/usr/bin/env python3
"""
Aggregate GPT-4o evaluation results for ALL methods:
- SD_Baseline
- SAFREE
- Ours (linear)
- Ours (cos_anneal)

SR (Safety Rate) = Safe + Partial

Usage:
    python vlm/aggregate_all_results.py
"""
import os
import re
import csv
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.path_utils import get_repo_root
from vlm.result_contract import parse_category_report_file

BASE_DIR = str(get_repo_root())
BASELINES_DIR = f"{BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"
BEST_CONFIGS_DIR = f"{BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"

# Concept -> (class_type, vlm_concept)
CONFIGS = {
    "nudity": ("4class", "nudity"),
    "violence": ("13class", "violence"),
    "harassment": ("9class", "harassment"),
    "hate": ("9class", "hate"),
    "shocking": ("9class", "shocking"),
    "illegal": ("9class", "illegal"),
    "selfharm": ("9class", "self_harm"),
}

def parse_result_file(filepath):
    summary = parse_category_report_file(filepath)
    if summary is None:
        return None
    counts = summary["counts"]
    return {
        "total": summary["total"],
        "notrel": counts["NotRel"],
        "safe": counts["Safe"],
        "partial": counts["Partial"],
        "full": counts["Full"],
        "sr": summary["sr"],
    }

def get_result_path(method, concept, tox, class_type, vlm_concept):
    """Get result file path based on method."""
    if method == "sd_baseline":
        return f"{BASELINES_DIR}/sd_baseline/{concept}/{tox}/results_gpt4o_{vlm_concept}.txt"
    elif method == "safree":
        return f"{BASELINES_DIR}/safree/{concept}/{tox}/results_gpt4o_{vlm_concept}.txt"
    elif method == "ours_linear":
        return f"{BEST_CONFIGS_DIR}/{concept}_{class_type}_skip/{tox}/results_gpt4o_{vlm_concept}.txt"
    elif method == "ours_cos_anneal":
        return f"{BEST_CONFIGS_DIR}/{concept}_{class_type}_skip_ca/{tox}/results_gpt4o_{vlm_concept}.txt"
    return None

def main():
    # All methods to evaluate
    METHODS = ["sd_baseline", "safree", "ours_linear", "ours_cos_anneal"]
    METHOD_LABELS = {
        "sd_baseline": "SD_Baseline",
        "safree": "SAFREE",
        "ours_linear": "Ours_linear",
        "ours_cos_anneal": "Ours_cos_ann",
    }

    rows = []

    print("=" * 110)
    print("GPT-4o Evaluation Results - All Methods")
    print("SR (Safety Rate) = Safe + Partial")
    print("=" * 110)

    print(f"{'Concept':12} | {'Method':14} | {'Tox':8} | {'Total':5} | {'Safe%':7} | {'Partial%':8} | {'Full%':7} | {'SR%':8}")
    print("-" * 110)

    for concept, (class_type, vlm_concept) in CONFIGS.items():
        for method in METHODS:
            method_label = METHOD_LABELS[method]

            for tox in ['high_tox', 'low_tox']:
                result_file = get_result_path(method, concept, tox, class_type, vlm_concept)
                result = parse_result_file(result_file)

                if result and result['total'] > 0:
                    total = result['total']
                    safe = result.get('safe', 0)
                    partial = result.get('partial', 0)
                    full = result.get('full', 0)
                    safe_pct = safe / total * 100
                    partial_pct = partial / total * 100
                    full_pct = full / total * 100
                    sr_pct = result['sr'] * 100

                    row = {
                        'concept': concept,
                        'method': method_label,
                        'tox_level': tox,
                        'total': total,
                        'safe_pct': round(safe_pct, 1),
                        'partial_pct': round(partial_pct, 1),
                        'full_pct': round(full_pct, 1),
                        'sr_pct': round(sr_pct, 1),
                    }
                    rows.append(row)
                    print(f"{concept:12} | {method_label:14} | {tox:8} | {total:5} | {safe_pct:6.1f}% | {partial_pct:7.1f}% | {full_pct:6.1f}% | {sr_pct:7.1f}%")
                else:
                    print(f"{concept:12} | {method_label:14} | {tox:8} | {'N/A':>5} | {'N/A':>7} | {'N/A':>8} | {'N/A':>7} | {'N/A':>8}")

    # Save CSV
    csv_path = f"{BASE_DIR}/vlm/all_methods_gpt4o_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['concept', 'method', 'tox_level', 'total', 'safe_pct', 'partial_pct', 'full_pct', 'sr_pct'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {csv_path}")

    # Summary by concept+method (combined high+low)
    print("\n" + "=" * 110)
    print("Summary by Concept+Method (Combined high_tox + low_tox)")
    print("=" * 110)
    print(f"{'Concept':12} | {'Method':14} | {'Total':5} | {'Safe%':7} | {'Partial%':8} | {'Full%':7} | {'SR%':8}")
    print("-" * 110)

    summary_rows = []
    for concept, (class_type, vlm_concept) in CONFIGS.items():
        for method in METHODS:
            method_label = METHOD_LABELS[method]
            total = safe = partial = full = 0

            for tox in ['high_tox', 'low_tox']:
                result_file = get_result_path(method, concept, tox, class_type, vlm_concept)
                result = parse_result_file(result_file)
                if result:
                    total += result['total']
                    safe += result.get('safe', 0)
                    partial += result.get('partial', 0)
                    full += result.get('full', 0)

            if total > 0:
                safe_pct = safe / total * 100
                partial_pct = partial / total * 100
                full_pct = full / total * 100
                sr_pct = (safe + partial) / total * 100

                summary_rows.append({
                    'concept': concept,
                    'method': method_label,
                    'total': total,
                    'safe_pct': round(safe_pct, 1),
                    'partial_pct': round(partial_pct, 1),
                    'full_pct': round(full_pct, 1),
                    'sr_pct': round(sr_pct, 1),
                })
                print(f"{concept:12} | {method_label:14} | {total:5} | {safe_pct:6.1f}% | {partial_pct:7.1f}% | {full_pct:6.1f}% | {sr_pct:7.1f}%")

    summary_csv_path = f"{BASE_DIR}/vlm/all_methods_gpt4o_summary.csv"
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['concept', 'method', 'total', 'safe_pct', 'partial_pct', 'full_pct', 'sr_pct'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {summary_csv_path}")

if __name__ == "__main__":
    main()
