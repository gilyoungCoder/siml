#!/usr/bin/env python3
"""
Aggregate NudeNet + GPT-4o evaluation results for nudity datasets.

Usage:
    python aggregate_nudity_results.py <dir1> <dir2> ...
    python aggregate_nudity_results.py /path/to/outputs/nudity_datasets/ringabell/*
    python aggregate_nudity_results.py --base-dir /path/to/outputs/nudity_datasets

Output:
    - Prints summary table
    - Saves CSV with all results
"""

import os
import re
import json
import argparse
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional
import csv


def parse_nudenet_result(result_file: str) -> Optional[Dict]:
    """Parse NudeNet results_nudenet.txt file."""
    if not os.path.exists(result_file):
        return None

    try:
        with open(result_file, 'r') as f:
            content = f.read()

        total_match = re.search(r'Total Images: (\d+)', content)
        unsafe_match = re.search(r'Unsafe Count: (\d+)', content)
        rate_match = re.search(r'Unsafe Rate: ([\d.]+)', content)

        total = int(total_match.group(1)) if total_match else 0
        unsafe = int(unsafe_match.group(1)) if unsafe_match else 0
        rate = float(rate_match.group(1)) if rate_match else 0.0

        return {
            'total': total,
            'unsafe_count': unsafe,
            'unsafe_rate': rate,
            'safe_rate': 1.0 - rate
        }
    except Exception as e:
        print(f"Error parsing {result_file}: {e}")
        return None


def parse_gpt_result(result_dir: str) -> Optional[Dict]:
    """Parse GPT-4o results from results.txt or categories_gpt4o_b64url.json."""
    # Try results.txt first
    results_txt = os.path.join(result_dir, "results_gpt4o.txt")
    categories_json = os.path.join(result_dir, "categories_gpt4o_b64url.json")

    if os.path.exists(results_txt):
        try:
            with open(results_txt, 'r') as f:
                content = f.read()

            # Try new format first: "Success (Safe+Partial): 65개 (82.3%)"
            success_match = re.search(r'Success \(Safe\+Partial\): (\d+)개 \(([\d.]+)%\)', content)
            harmful_match = re.search(r'Harmful \(Full\): (\d+)개', content)
            total_match = re.search(r'Total: (\d+)', content)

            if success_match and total_match:
                # New format
                success_count = int(success_match.group(1))
                success_rate = float(success_match.group(2)) / 100
                total = int(total_match.group(1))
                full = int(harmful_match.group(1)) if harmful_match else 0

                # Parse individual counts
                safe_match = re.search(r'Safe: (\d+)개', content)
                partial_match = re.search(r'Partial: (\d+)개', content)
                not_people_match = re.search(r'NotPeople: (\d+)개', content)

                safe = int(safe_match.group(1)) if safe_match else 0
                partial = int(partial_match.group(1)) if partial_match else 0
                not_people = int(not_people_match.group(1)) if not_people_match else 0

                return {
                    'total': total,
                    'safe': safe,
                    'partial': partial,
                    'full': full,
                    'not_people': not_people,
                    'success_count': success_count,
                    'success_rate': success_rate
                }
            else:
                # Old format fallback
                counts = {}
                for line in content.split('\n'):
                    match = re.search(r'(\w+): (\d+)개', line)
                    if match:
                        counts[match.group(1)] = int(match.group(2))

                safe = counts.get('Safe', 0)
                partial = counts.get('Partial', 0)
                full = counts.get('Full', 0)
                not_people = counts.get('NotPeople', 0)
                total = safe + partial + full + not_people

                if total > 0:
                    success_rate = (safe + partial) / total
                else:
                    success_rate = 0.0

                return {
                    'total': total,
                    'safe': safe,
                    'partial': partial,
                    'full': full,
                    'not_people': not_people,
                    'success_count': safe + partial,
                    'success_rate': success_rate
                }
        except Exception as e:
            print(f"Error parsing {results_txt}: {e}")

    # Try JSON file
    if os.path.exists(categories_json):
        try:
            with open(categories_json, 'r') as f:
                data = json.load(f)

            counts = {'Safe': 0, 'Partial': 0, 'Full': 0, 'NotPeople': 0, 'Error': 0}
            for filename, result in data.items():
                cat = result.get('category', 'Error')
                if cat in counts:
                    counts[cat] += 1
                else:
                    counts['Error'] += 1

            safe = counts['Safe']
            partial = counts['Partial']
            full = counts['Full']
            not_people = counts['NotPeople']
            total = safe + partial + full + not_people

            if total > 0:
                success_rate = (safe + partial) / total
            else:
                success_rate = 0.0

            return {
                'total': total,
                'safe': safe,
                'partial': partial,
                'full': full,
                'not_people': not_people,
                'success_count': safe + partial,
                'success_rate': success_rate
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
        nudenet_file = os.path.join(actual_path, "results_nudenet.txt")
        nudenet = parse_nudenet_result(nudenet_file)
        if nudenet:
            result['nudenet_total'] = nudenet['total']
            result['nudenet_unsafe'] = nudenet['unsafe_count']
            result['nudenet_unsafe_rate'] = nudenet['unsafe_rate']
            result['nudenet_safe_rate'] = nudenet['safe_rate']
        else:
            result['nudenet_total'] = None
            result['nudenet_unsafe'] = None
            result['nudenet_unsafe_rate'] = None
            result['nudenet_safe_rate'] = None

        # Parse GPT-4o results
        gpt = parse_gpt_result(actual_path)
        if gpt:
            result['gpt_total'] = gpt['total']
            result['gpt_safe'] = gpt['safe']
            result['gpt_partial'] = gpt['partial']
            result['gpt_full'] = gpt['full']
            result['gpt_not_people'] = gpt['not_people']
            result['gpt_success_count'] = gpt['success_count']
            result['gpt_success_rate'] = gpt['success_rate']
        else:
            result['gpt_total'] = None
            result['gpt_safe'] = None
            result['gpt_partial'] = None
            result['gpt_full'] = None
            result['gpt_not_people'] = None
            result['gpt_success_count'] = None
            result['gpt_success_rate'] = None

        results.append(result)

    return results


def print_summary_table(results: List[Dict]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("NUDITY EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\n{'Directory':<35} {'NudeNet%':>10} {'Safe':>6} {'Partial':>8} {'Full':>6} {'Success%':>10}")
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

            # GPT-4o
            if r['gpt_success_rate'] is not None:
                safe_str = str(r['gpt_safe'])
                partial_str = str(r['gpt_partial'])
                full_str = str(r['gpt_full'])
                success_str = f"{r['gpt_success_rate']*100:.1f}%"
            else:
                safe_str = "-"
                partial_str = "-"
                full_str = "-"
                success_str = "missing"

            print(f"  {name:<33} {nudenet_str:>10} {safe_str:>6} {partial_str:>8} {full_str:>6} {success_str:>10}")

    print("\n" + "=" * 100)
    print("Legend:")
    print("  NudeNet%  - Unsafe rate by NudeNet classifier (lower is better)")
    print("  Success%  - (Safe + Partial) / Total by GPT-4o (higher is better)")
    print("=" * 100)


def save_csv(results: List[Dict], output_path: str):
    """Save results to CSV."""
    if not results:
        return

    fieldnames = [
        'dataset', 'name', 'full_name',
        'nudenet_total', 'nudenet_unsafe', 'nudenet_unsafe_rate', 'nudenet_safe_rate',
        'gpt_total', 'gpt_safe', 'gpt_partial', 'gpt_full', 'gpt_not_people',
        'gpt_success_count', 'gpt_success_rate',
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
    parser = argparse.ArgumentParser(description="Aggregate NudeNet + GPT-4o evaluation results")
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
