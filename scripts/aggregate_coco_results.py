#!/usr/bin/env python3
"""
Aggregate COCO FID + CLIP evaluation results.

Usage:
    python aggregate_coco_results.py --base-dir /path/to/outputs/coco
    python aggregate_coco_results.py /path/to/method1 /path/to/method2 ...
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def parse_metrics(result_dir: str) -> Optional[Dict]:
    """Parse eval_metrics.json from a directory."""
    # Check if results are in 'generated' subfolder
    generated_path = os.path.join(result_dir, "generated")
    if os.path.isdir(generated_path):
        actual_path = generated_path
    else:
        actual_path = result_dir

    metrics_file = os.path.join(actual_path, "eval_metrics.json")
    if not os.path.exists(metrics_file):
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return {
            'fid': data.get('fid'),
            'clip_score': data.get('clip_score'),
            'n_images': data.get('n_images')
        }
    except Exception as e:
        print(f"Error parsing {metrics_file}: {e}")
        return None


def aggregate_results(directories: List[str]) -> List[Dict]:
    """Aggregate results from multiple directories."""
    results = []

    for dir_path in directories:
        if not os.path.isdir(dir_path):
            continue

        dir_name = os.path.basename(dir_path)
        parent_name = os.path.basename(os.path.dirname(dir_path))

        result = {
            'directory': dir_path,
            'name': dir_name,
            'dataset': parent_name,
        }

        metrics = parse_metrics(dir_path)
        if metrics:
            result['fid'] = metrics['fid']
            result['clip_score'] = metrics['clip_score']
            result['n_images'] = metrics['n_images']
        else:
            result['fid'] = None
            result['clip_score'] = None
            result['n_images'] = None

        results.append(result)

    return results


def print_summary_table(results: List[Dict]):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("COCO EVALUATION SUMMARY (FID + CLIP)")
    print("=" * 70)
    print(f"\n{'Method':<30} {'Images':>8} {'FID':>12} {'CLIP':>12}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['name']):
        name = r['name'][:28]
        n_images = str(r['n_images']) if r['n_images'] else "-"

        if r['fid'] is not None:
            fid_str = f"{r['fid']:.2f}"
        else:
            fid_str = "missing"

        if r['clip_score'] is not None:
            clip_str = f"{r['clip_score']:.4f}"
        else:
            clip_str = "missing"

        print(f"  {name:<28} {n_images:>8} {fid_str:>12} {clip_str:>12}")

    print("\n" + "=" * 70)
    print("Legend:")
    print("  FID  - Frechet Inception Distance (lower is better)")
    print("  CLIP - CLIP Score (higher is better)")
    print("=" * 70)


def save_csv(results: List[Dict], output_path: str):
    """Save results to CSV."""
    import csv

    fieldnames = ['name', 'n_images', 'fid', 'clip_score', 'directory']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate COCO FID + CLIP results")
    parser.add_argument('directories', nargs='*', help='Directories to aggregate')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory containing method subdirectories')
    parser.add_argument('--output', type=str, default='coco_eval_summary.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    directories = []

    # If base-dir is specified, find all method subdirectories
    if args.base_dir:
        base = Path(args.base_dir)
        if base.exists():
            for method_dir in base.iterdir():
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
