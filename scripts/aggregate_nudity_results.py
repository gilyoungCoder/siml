#!/usr/bin/env python3
"""
Aggregate NudeNet + nudity-judge evaluation results for nudity datasets.

Usage:
    python aggregate_nudity_results.py <dir1> <dir2> ...
    python aggregate_nudity_results.py /path/to/outputs/nudity_datasets/ringabell/*
    python aggregate_nudity_results.py --base-dir /path/to/outputs/nudity_datasets

Output:
    - Prints summary table
    - Saves CSV with all results
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import csv
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.result_contract import (
    load_category_json_summary,
    parse_category_report_file,
    parse_nudenet_result_file,
)
from vlm.result_paths import (
    categories_json_candidates,
    find_existing_result_file,
    results_txt_candidates,
)


def parse_nudenet_result(result_file: str) -> Optional[Dict]:
    """Parse NudeNet results_nudenet.txt file."""
    return parse_nudenet_result_file(result_file)


def find_nudenet_result_file(result_dir: str) -> Optional[str]:
    for name in ("results_nudenet.txt", "results_nudenet_08.txt", "results_nudenet_06.txt"):
        path = os.path.join(result_dir, name)
        if os.path.exists(path):
            return path
    return None


def parse_judge_result(result_dir: str) -> Optional[Dict]:
    """Parse current/legacy Qwen nudity-judge results from a result directory."""
    results_txt = find_existing_result_file(
        result_dir,
        results_txt_candidates("qwen", "nudity") + ["results_gpt4o.txt"],
    )
    categories_json = find_existing_result_file(
        result_dir,
        categories_json_candidates("qwen", "nudity") + ["categories_gpt4o_b64url.json"],
    )

    if results_txt is not None:
        try:
            summary = parse_category_report_file(results_txt)
            if summary is not None:
                counts = summary['counts']
                return {
                    'result_file': str(results_txt),
                    'json_file': str(categories_json) if categories_json is not None else None,
                    'total': summary['total'],
                    'safe': counts['Safe'],
                    'partial': counts['Partial'],
                    'full': counts['Full'],
                    'not_people': counts['NotRel'],
                    'success_count': summary['safe_count'],
                    'success_rate': summary['sr'],
                    'relevant_success_rate': summary['relevant_sr'],
                }
        except Exception as e:
            print(f"Error parsing {results_txt}: {e}")

    # Try JSON file
    if categories_json is not None:
        try:
            summary = load_category_json_summary(categories_json)
            counts = summary['counts']
            return {
                'result_file': None,
                'json_file': str(categories_json),
                'total': summary['total'],
                'safe': counts['Safe'],
                'partial': counts['Partial'],
                'full': counts['Full'],
                'not_people': counts['NotRel'],
                'success_count': summary['safe_count'],
                'success_rate': summary['sr'],
                'relevant_success_rate': summary['relevant_sr'],
            }
        except Exception as e:
            print(f"Error parsing {categories_json}: {e}")

    return None


def aggregate_results(directories: List[str]) -> List[Dict]:
    """Aggregate results from multiple directories."""
    results = []

    for dir_path in directories:
        if not os.path.isdir(dir_path):
            continue

        dir_name = os.path.basename(dir_path)
        parent_name = os.path.basename(os.path.dirname(dir_path))

        # Check if results are in 'generated' subfolder (for safree_regenerated)
        generated_path = os.path.join(dir_path, "generated")
        if os.path.isdir(generated_path):
            actual_path = generated_path
        else:
            actual_path = dir_path

        result = {
            'directory': dir_path,
            'name': dir_name,
            'dataset': parent_name,
            'full_name': f"{parent_name}/{dir_name}"
        }

        # Parse NudeNet results
        nudenet_file = find_nudenet_result_file(actual_path)
        nudenet = parse_nudenet_result(nudenet_file) if nudenet_file else None
        if nudenet:
            result['nudenet_file'] = nudenet_file
            result['nudenet_total'] = nudenet['total']
            result['nudenet_unsafe'] = nudenet['unsafe_count']
            result['nudenet_unsafe_rate'] = nudenet['unsafe_rate']
            result['nudenet_safe_rate'] = nudenet['safe_rate']
        else:
            result['nudenet_file'] = None
            result['nudenet_total'] = None
            result['nudenet_unsafe'] = None
            result['nudenet_unsafe_rate'] = None
            result['nudenet_safe_rate'] = None

        # Parse Qwen / judge results
        judge = parse_judge_result(actual_path)
        if judge:
            result['qwen_result_file'] = judge['result_file']
            result['qwen_json_file'] = judge['json_file']
            result['qwen_total'] = judge['total']
            result['qwen_safe'] = judge['safe']
            result['qwen_partial'] = judge['partial']
            result['qwen_full'] = judge['full']
            result['qwen_not_people'] = judge['not_people']
            result['qwen_success_count'] = judge['success_count']
            result['qwen_success_rate'] = judge['success_rate']
            result['qwen_relevant_success_rate'] = judge['relevant_success_rate']
        else:
            result['qwen_result_file'] = None
            result['qwen_json_file'] = None
            result['qwen_total'] = None
            result['qwen_safe'] = None
            result['qwen_partial'] = None
            result['qwen_full'] = None
            result['qwen_not_people'] = None
            result['qwen_success_count'] = None
            result['qwen_success_rate'] = None
            result['qwen_relevant_success_rate'] = None

        results.append(result)

    return results


def print_summary_table(results: List[Dict]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("NUDITY EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\n{'Directory':<35} {'NudeNet%':>10} {'Safe':>6} {'Partial':>8} {'Full':>6} {'SR%':>10}")
    print("-" * 100)

    # Group by dataset
    datasets = {}
    for r in results:
        dataset = r['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(r)

    for dataset, items in sorted(datasets.items()):
        print(f"\n[{dataset}]")
        for r in sorted(items, key=lambda x: x['name']):
            name = r['name'][:32]

            # NudeNet
            if r['nudenet_unsafe_rate'] is not None:
                nudenet_str = f"{r['nudenet_unsafe_rate']*100:.1f}%"
            else:
                nudenet_str = "missing"

            # Qwen / judge
            if r['qwen_success_rate'] is not None:
                safe_str = str(r['qwen_safe'])
                partial_str = str(r['qwen_partial'])
                full_str = str(r['qwen_full'])
                success_str = f"{r['qwen_success_rate']*100:.1f}%"
            else:
                safe_str = "-"
                partial_str = "-"
                full_str = "-"
                success_str = "missing"

            print(f"  {name:<33} {nudenet_str:>10} {safe_str:>6} {partial_str:>8} {full_str:>6} {success_str:>10}")

    print("\n" + "=" * 100)
    print("Legend:")
    print("  NudeNet%  - Unsafe rate by NudeNet classifier (lower is better)")
    print("  SR%       - (Safe + Partial) / Total by the nudity judge (higher is better)")
    print("=" * 100)


def save_csv(results: List[Dict], output_path: str):
    """Save results to CSV."""
    if not results:
        return

    fieldnames = [
        'dataset', 'name', 'full_name',
        'nudenet_file', 'nudenet_total', 'nudenet_unsafe', 'nudenet_unsafe_rate', 'nudenet_safe_rate',
        'qwen_result_file', 'qwen_json_file',
        'qwen_total', 'qwen_safe', 'qwen_partial', 'qwen_full', 'qwen_not_people',
        'qwen_success_count', 'qwen_success_rate', 'qwen_relevant_success_rate',
        'directory'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate NudeNet + nudity-judge evaluation results")
    parser.add_argument('directories', nargs='*', help='Directories to aggregate')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory containing dataset subdirectories')
    parser.add_argument('--output', type=str, default='nudity_eval_summary.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    directories = []

    # If base-dir is specified, find all method subdirectories
    if args.base_dir:
        base = Path(args.base_dir)
        if base.exists():
            # Pattern: base_dir/dataset/method/
            for dataset_dir in base.iterdir():
                if dataset_dir.is_dir():
                    for method_dir in dataset_dir.iterdir():
                        if method_dir.is_dir():
                            directories.append(str(method_dir))

    # Add explicitly specified directories
    directories.extend(args.directories)

    if not directories:
        print("ERROR: No directories specified. Use positional args or --base-dir")
        return

    # Remove duplicates and sort
    directories = sorted(set(directories))
    print(f"Found {len(directories)} directories to aggregate")

    # Aggregate results
    results = aggregate_results(directories)

    if not results:
        print("No results found")
        return

    # Print summary
    print_summary_table(results)

    # Save CSV
    save_csv(results, args.output)


if __name__ == '__main__':
    main()
