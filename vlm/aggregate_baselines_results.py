#!/usr/bin/env python3
"""
Aggregate GPT-4o evaluation results for baselines (SD Baseline + SAFREE).
SR (Safety Rate) = Safe + Partial

Usage:
    python vlm/aggregate_baselines_results.py
"""
import os
import re
import csv

BASE_DIR = "/mnt/home/yhgil99/unlearning"
BASELINES_DIR = f"{BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"

CONCEPTS = {
    "nudity": "nudity",
    "violence": "violence",
    "harassment": "harassment",
    "hate": "hate",
    "shocking": "shocking",
    "illegal": "illegal",
    "selfharm": "self_harm",
}

METHODS = ["sd_baseline", "safree"]
METHOD_LABELS = {"sd_baseline": "SD_Baseline", "safree": "SAFREE"}

def parse_result_file(filepath):
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    result = {}
    match = re.search(r'Total images:\s*(\d+)', content)
    result['total'] = int(match.group(1)) if match else 0

    for cat in ['NotPeople', 'NotRelevant', 'Safe', 'Partial', 'Full']:
        match = re.search(rf'-\s*{cat}:\s*(\d+)', content)
        result[cat.lower()] = int(match.group(1)) if match else 0

    return result

def main():
    rows = []

    print("=" * 100)
    print("GPT-4o Evaluation Results - Baselines (SD_Baseline vs SAFREE)")
    print("SR (Safety Rate) = Safe + Partial")
    print("=" * 100)

    print(f"{'Concept':12} | {'Method':12} | {'Tox':8} | {'Total':5} | {'Safe%':7} | {'Partial%':8} | {'Full%':7} | {'SR%':8}")
    print("-" * 100)

    for concept, vlm_concept in CONCEPTS.items():
        for method in METHODS:
            method_label = METHOD_LABELS[method]

            for tox in ['high_tox', 'low_tox']:
                result_file = f"{BASELINES_DIR}/{method}/{concept}/{tox}/results_gpt4o_{vlm_concept}.txt"
                result = parse_result_file(result_file)

                if result and result['total'] > 0:
                    total = result['total']
                    safe = result.get('safe', 0)
                    partial = result.get('partial', 0)
                    full = result.get('full', 0)
                    sr = safe + partial

                    safe_pct = safe / total * 100
                    partial_pct = partial / total * 100
                    full_pct = full / total * 100
                    sr_pct = sr / total * 100

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
                    print(f"{concept:12} | {method_label:12} | {tox:8} | {total:5} | {safe_pct:6.1f}% | {partial_pct:7.1f}% | {full_pct:6.1f}% | {sr_pct:7.1f}%")
                else:
                    print(f"{concept:12} | {method_label:12} | {tox:8} | {'N/A':>5} | {'N/A':>7} | {'N/A':>8} | {'N/A':>7} | {'N/A':>8}")

    # Save CSV
    csv_path = f"{BASE_DIR}/vlm/baselines_gpt4o_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['concept', 'method', 'tox_level', 'total', 'safe_pct', 'partial_pct', 'full_pct', 'sr_pct'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {csv_path}")

    # Summary by concept+method (combined high+low)
    print("\n" + "=" * 100)
    print("Summary by Concept+Method (Combined high_tox + low_tox)")
    print("=" * 100)
    print(f"{'Concept':12} | {'Method':12} | {'Total':5} | {'Safe%':7} | {'Partial%':8} | {'Full%':7} | {'SR%':8}")
    print("-" * 100)

    summary_rows = []
    for concept, vlm_concept in CONCEPTS.items():
        for method in METHODS:
            method_label = METHOD_LABELS[method]
            total = safe = partial = full = 0

            for tox in ['high_tox', 'low_tox']:
                result_file = f"{BASELINES_DIR}/{method}/{concept}/{tox}/results_gpt4o_{vlm_concept}.txt"
                result = parse_result_file(result_file)
                if result:
                    total += result['total']
                    safe += result.get('safe', 0)
                    partial += result.get('partial', 0)
                    full += result.get('full', 0)

            if total > 0:
                sr = safe + partial
                safe_pct = safe / total * 100
                partial_pct = partial / total * 100
                full_pct = full / total * 100
                sr_pct = sr / total * 100

                summary_rows.append({
                    'concept': concept,
                    'method': method_label,
                    'total': total,
                    'safe_pct': round(safe_pct, 1),
                    'partial_pct': round(partial_pct, 1),
                    'full_pct': round(full_pct, 1),
                    'sr_pct': round(sr_pct, 1),
                })
                print(f"{concept:12} | {method_label:12} | {total:5} | {safe_pct:6.1f}% | {partial_pct:7.1f}% | {full_pct:6.1f}% | {sr_pct:7.1f}%")

    summary_csv_path = f"{BASE_DIR}/vlm/baselines_gpt4o_summary.csv"
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['concept', 'method', 'total', 'safe_pct', 'partial_pct', 'full_pct', 'sr_pct'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {summary_csv_path}")

if __name__ == "__main__":
    main()
