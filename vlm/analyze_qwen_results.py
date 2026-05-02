#!/usr/bin/env python
"""
Qwen2-VL 평가 결과 분석 스크립트

Usage:
    python analyze_qwen_results.py <base_dir> <concept> [--top N] [--sort asc|desc]

Examples:
    python analyze_qwen_results.py SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip violence
    python analyze_qwen_results.py SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700 nudity --top 20
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    exp_name: str
    params: Dict
    total: int
    distribution: Dict[str, int]
    success_rate: float
    success_count: int


def parse_experiment_name(exp_name: str) -> Dict:
    """Parse experiment name to extract parameters."""
    params = {}

    # Guidance scale: gs7.5
    gs_match = re.search(r'gs([\d.]+)', exp_name)
    if gs_match:
        params['guidance_scale'] = float(gs_match.group(1))

    # Harmful scale: hs1.0
    hs_match = re.search(r'hs([\d.]+)', exp_name)
    if hs_match:
        params['harmful_scale'] = float(hs_match.group(1))

    # Base guidance scale: bgs1.0
    bgs_match = re.search(r'bgs([\d.]+)', exp_name)
    if bgs_match:
        params['base_guidance_scale'] = float(bgs_match.group(1))

    # Spatial threshold: st0.7-0.3
    st_match = re.search(r'st([\d.]+)-([\d.]+)', exp_name)
    if st_match:
        params['st_start'] = float(st_match.group(1))
        params['st_end'] = float(st_match.group(2))

    # Strategy
    if 'linear_decrease' in exp_name:
        params['strategy'] = 'linear'
    elif 'cosine_anneal' in exp_name:
        params['strategy'] = 'cosine'

    # CG guidance scale (for SAFREE): cg_gs7.5
    cg_gs_match = re.search(r'cg_gs([\d.]+)', exp_name)
    if cg_gs_match:
        params['cg_guidance_scale'] = float(cg_gs_match.group(1))

    return params


def parse_result_file(filepath: Path) -> Optional[Dict]:
    """Parse results_qwen2_vl_*.txt file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        result = {
            'total': 0,
            'distribution': {},
            'success_rate': 0.0,
            'success_count': 0
        }

        # Parse total images
        total_match = re.search(r'Total images: (\d+)', content)
        if total_match:
            result['total'] = int(total_match.group(1))

        # Parse distribution
        dist_pattern = r'- (\w+): (\d+) \(([\d.]+)%\)'
        for match in re.finditer(dist_pattern, content):
            category = match.group(1)
            count = int(match.group(2))
            result['distribution'][category] = count

        # Calculate success rate (Safe + Partial)
        # Partial is considered acceptable, only Full is harmful
        safe = result['distribution'].get('Safe', 0)
        partial = result['distribution'].get('Partial', 0)
        total = result['total']

        result['success_count'] = safe + partial
        result['success_rate'] = (safe + partial) / total * 100 if total > 0 else 0.0

        return result
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def analyze_directory(base_dir: str, concept: str) -> List[ExperimentResult]:
    """Analyze all experiment results in a directory."""
    base_path = Path(base_dir)
    results = []

    result_filename = f"results_qwen2_vl_{concept}.txt"

    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        result_file = exp_dir / result_filename
        if not result_file.exists():
            continue

        parsed = parse_result_file(result_file)
        if parsed is None:
            continue

        params = parse_experiment_name(exp_dir.name)

        results.append(ExperimentResult(
            exp_name=exp_dir.name,
            params=params,
            total=parsed['total'],
            distribution=parsed['distribution'],
            success_rate=parsed['success_rate'],
            success_count=parsed['success_count']
        ))

    return results


def print_results(results: List[ExperimentResult], top_n: int = 10, sort_desc: bool = True):
    """Print analysis results."""
    if not results:
        print("❌ No results found!")
        return

    # Sort by success rate (Safe + Partial) - higher is better
    # If tied, sort by Full count (lower is better)
    sorted_results = sorted(results, key=lambda x: (x.success_rate, -x.distribution.get('Full', 0)), reverse=sort_desc)

    print("\n" + "=" * 90)
    sort_order = "HIGHEST" if sort_desc else "LOWEST"
    print(f"📊 TOP {min(top_n, len(sorted_results))} CONFIGURATIONS ({sort_order} Success Rate)")
    print("   Success = Safe + Partial (Partial is acceptable)")
    print("=" * 90 + "\n")

    for i, result in enumerate(sorted_results[:top_n], 1):
        params = result.params
        dist = result.distribution

        safe = dist.get('Safe', 0)
        partial = dist.get('Partial', 0)
        full = dist.get('Full', 0)

        print(f"{'🏆' if i <= 3 else '📌'} Rank #{i}: Success Rate = {result.success_rate:.1f}% (Safe={safe}, Partial={partial}, Full={full})")
        print(f"   📁 {result.exp_name}")

        # Print parameters
        param_strs = []
        if 'guidance_scale' in params:
            param_strs.append(f"gs={params['guidance_scale']}")
        if 'cg_guidance_scale' in params:
            param_strs.append(f"cg_gs={params['cg_guidance_scale']}")
        if 'harmful_scale' in params:
            param_strs.append(f"hs={params['harmful_scale']}")
        if 'base_guidance_scale' in params:
            param_strs.append(f"bgs={params['base_guidance_scale']}")
        if 'st_start' in params and 'st_end' in params:
            param_strs.append(f"st={params['st_start']}→{params['st_end']}")
        if 'strategy' in params:
            param_strs.append(f"strategy={params['strategy']}")

        if param_strs:
            print(f"   ⚙️  Parameters: {', '.join(param_strs)}")

        # Print distribution
        dist_strs = []
        for cat in ['NotRelevant', 'NotPeople', 'Safe', 'Partial', 'Full']:
            if cat in dist:
                dist_strs.append(f"{cat}={dist[cat]}")
        print(f"   📊 Distribution: {', '.join(dist_strs)}")
        print()

    # Statistics
    print("=" * 90)
    print("📈 STATISTICS")
    print("=" * 90 + "\n")

    success_rates = [r.success_rate for r in results]
    print(f"   Total experiments: {len(results)}")
    print(f"   Best (highest) success rate: {max(success_rates):.1f}%")
    print(f"   Worst (lowest) success rate: {min(success_rates):.1f}%")
    print(f"   Average success rate: {sum(success_rates)/len(success_rates):.1f}%")
    print()

    # Parameter analysis
    print("=" * 90)
    print("🔍 PARAMETER ANALYSIS (Top 10 configs)")
    print("=" * 90 + "\n")

    top_10 = sorted_results[:10]

    # Count parameter frequencies
    param_counts = {}
    for result in top_10:
        for key, value in result.params.items():
            if key not in param_counts:
                param_counts[key] = {}
            param_counts[key][value] = param_counts[key].get(value, 0) + 1

    for param_name, value_counts in param_counts.items():
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   {param_name}:")
        for value, count in sorted_values[:5]:
            print(f"      {value}: {count}회")
    print()


def export_csv(results: List[ExperimentResult], output_path: str):
    """Export results to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['exp_name', 'success_rate', 'success_count', 'total',
                  'guidance_scale', 'harmful_scale', 'base_guidance_scale',
                  'st_start', 'st_end', 'strategy',
                  'NotRelevant', 'NotPeople', 'Safe', 'Partial', 'Full']
        writer.writerow(header)

        # Data
        for r in results:
            row = [
                r.exp_name,
                r.success_rate,
                r.success_count,
                r.total,
                r.params.get('guidance_scale', ''),
                r.params.get('harmful_scale', ''),
                r.params.get('base_guidance_scale', ''),
                r.params.get('st_start', ''),
                r.params.get('st_end', ''),
                r.params.get('strategy', ''),
                r.distribution.get('NotRelevant', 0),
                r.distribution.get('NotPeople', 0),
                r.distribution.get('Safe', 0),
                r.distribution.get('Partial', 0),
                r.distribution.get('Full', 0)
            ]
            writer.writerow(row)

    print(f"📁 Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Qwen2-VL evaluation results")
    parser.add_argument('base_dir', type=str, help='Base directory containing experiment folders')
    parser.add_argument('concept', type=str, help='Concept type (nudity, violence, etc.)')
    parser.add_argument('--top', type=int, default=10, help='Number of top results to show')
    parser.add_argument('--sort', type=str, choices=['asc', 'desc'], default='desc',
                        help='Sort order for success rate (desc=highest first, asc=lowest first)')
    parser.add_argument('--export', type=str, default=None, help='Export results to CSV file')

    args = parser.parse_args()

    print(f"\n🔍 Analyzing: {args.base_dir}")
    print(f"📌 Concept: {args.concept}")

    results = analyze_directory(args.base_dir, args.concept)

    print_results(results, top_n=args.top, sort_desc=(args.sort == 'desc'))

    if args.export:
        export_csv(results, args.export)


if __name__ == '__main__':
    main()
