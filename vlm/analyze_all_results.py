#!/usr/bin/env python
"""
전체 Qwen2-VL 평가 결과 요약 분석

Usage:
    python analyze_all_results.py [--export results_summary.csv]
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import csv


@dataclass
class ExperimentResult:
    exp_name: str
    full_path: str
    success_rate: float
    full: int
    safe: int
    partial: int
    params: Dict


@dataclass
class ConceptSummary:
    name: str
    folder: str
    total_experiments: int
    best_success_rate: float
    avg_success_rate: float
    worst_success_rate: float
    best_config: str
    best_params: Dict
    best_full: int  # Full count of the best config
    best_safe: int
    best_partial: int
    top_results: List[ExperimentResult]  # Top N results with full paths


def parse_experiment_name(exp_name: str) -> Dict:
    """Parse experiment name to extract parameters."""
    params = {}

    gs_match = re.search(r'gs([\d.]+)', exp_name)
    if gs_match:
        params['gs'] = float(gs_match.group(1))

    hs_match = re.search(r'hs([\d.]+)', exp_name)
    if hs_match:
        params['hs'] = float(hs_match.group(1))

    bgs_match = re.search(r'bgs([\d.]+)', exp_name)
    if bgs_match:
        params['bgs'] = float(bgs_match.group(1))

    st_match = re.search(r'st([\d.]+)-([\d.]+)', exp_name)
    if st_match:
        params['st'] = f"{st_match.group(1)}→{st_match.group(2)}"

    cgs_match = re.search(r'cgs([\d.]+)', exp_name)
    if cgs_match:
        params['cgs'] = float(cgs_match.group(1))

    if 'linear_decrease' in exp_name:
        params['strategy'] = 'linear'
    elif 'cosine_anneal' in exp_name:
        params['strategy'] = 'cosine'

    return params


def parse_result_file(filepath: Path) -> Optional[Dict]:
    """Parse results_qwen2_vl_*.txt file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Parse total
        total_match = re.search(r'Total images: (\d+)', content)
        total = int(total_match.group(1)) if total_match else 0

        # Parse distribution
        dist = {}
        dist_pattern = r'- (\w+): (\d+) \(([\d.]+)%\)'
        for match in re.finditer(dist_pattern, content):
            dist[match.group(1)] = int(match.group(2))

        # Calculate success rate (Safe + Partial)
        safe = dist.get('Safe', 0)
        partial = dist.get('Partial', 0)
        full = dist.get('Full', 0)
        success_rate = (safe + partial) / total * 100 if total > 0 else 0.0

        return {
            'success_count': safe + partial,
            'total': total,
            'success_rate': success_rate,
            'full': full,
            'safe': safe,
            'partial': partial
        }
    except:
        return None


def analyze_folder(base_dir: str, concept: str, top_n: int = 5) -> Optional[ConceptSummary]:
    """Analyze a single folder."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    result_filename = f"results_qwen2_vl_{concept}.txt"
    results = []

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        result_file = exp_dir / result_filename
        if not result_file.exists():
            continue

        parsed = parse_result_file(result_file)
        if parsed:
            results.append({
                'exp_name': exp_dir.name,
                'full_path': str(exp_dir),
                'success_rate': parsed['success_rate'],
                'full': parsed['full'],
                'safe': parsed['safe'],
                'partial': parsed['partial'],
                'params': parse_experiment_name(exp_dir.name)
            })

    if not results:
        return None

    # Sort by success rate (higher is better), then by Full count (lower is better)
    sorted_results = sorted(results, key=lambda x: (x['success_rate'], -x['full']), reverse=True)
    best = sorted_results[0]

    rates = [r['success_rate'] for r in results]

    # Create top N ExperimentResult objects
    top_results = []
    for r in sorted_results[:top_n]:
        top_results.append(ExperimentResult(
            exp_name=r['exp_name'],
            full_path=r['full_path'],
            success_rate=r['success_rate'],
            full=r['full'],
            safe=r['safe'],
            partial=r['partial'],
            params=r['params']
        ))

    return ConceptSummary(
        name=concept,
        folder=str(base_path),
        total_experiments=len(results),
        best_success_rate=max(rates),
        avg_success_rate=sum(rates) / len(rates),
        worst_success_rate=min(rates),
        best_config=best['exp_name'],
        best_params=best['params'],
        best_full=best['full'],
        best_safe=best['safe'],
        best_partial=best['partial'],
        top_results=top_results
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze all Qwen2-VL evaluation results")
    parser.add_argument('--export', type=str, default=None, help='Export to CSV')
    args = parser.parse_args()

    base_unlearning = "/mnt/home/yhgil99/unlearning"

    # Define all folders to analyze
    folders_to_analyze = [
        # SCG 9-class (grid_search_results)
        ("SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300", "harassment"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip", "harassment"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800", "hate"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip", "hate"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600", "illegal"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip", "illegal"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700", "self_harm"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip", "self_harm"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700", "shocking"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip", "shocking"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500", "violence"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip", "violence"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class", "nudity"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always", "nudity"),

        # Violence 13-class
        ("SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400", "violence"),
        ("SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip", "violence"),

        # Nudity 6-class
        ("SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700", "nudity"),
        ("SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip", "nudity"),

        # SAFREE 9-class
        ("SAFREE/results/grid_search_safree_9class/harassment_step24300", "harassment"),
        ("SAFREE/results/grid_search_safree_9class/hate_step20800", "hate"),
        ("SAFREE/results/grid_search_safree_9class/illegal_step22600", "illegal"),
        ("SAFREE/results/grid_search_safree_9class/selfharm_step20700", "self_harm"),
        ("SAFREE/results/grid_search_safree_9class/shocking_step23700", "shocking"),
        ("SAFREE/results/grid_search_safree_9class/violence_step15500", "violence"),

        # SAFREE 4-class nudity
        ("SAFREE/results/grid_search_safree_4class_nudity", "nudity"),
    ]

    print("\n" + "=" * 100)
    print("📊 COMPREHENSIVE VLM EVALUATION RESULTS SUMMARY")
    print("=" * 100)

    all_summaries = []

    for rel_path, concept in folders_to_analyze:
        full_path = os.path.join(base_unlearning, rel_path)
        summary = analyze_folder(full_path, concept)

        if summary and summary.total_experiments > 0:
            all_summaries.append(summary)

    # Group by concept type
    concept_groups = {}
    for s in all_summaries:
        folder_name = Path(s.folder).name
        key = f"{s.name} ({folder_name})"
        concept_groups[key] = s

    # Print results table
    print(f"\n{'Folder':<60} {'Exp':>5} {'Best':>8} {'Avg':>8} {'Worst':>8}")
    print("   (Success Rate = Safe + Partial, higher is better)")
    print("-" * 100)

    for key in sorted(concept_groups.keys()):
        s = concept_groups[key]
        folder_short = Path(s.folder).name[:55]
        print(f"{folder_short:<60} {s.total_experiments:>5} {s.best_success_rate:>7.1f}% {s.avg_success_rate:>7.1f}% {s.worst_success_rate:>7.1f}%")

    # Print top 5 configs with full paths
    print("\n" + "=" * 100)
    print("🏆 TOP 5 CONFIGURATIONS BY CONCEPT (Highest Success Rate)")
    print("   Sorted by: Success Rate (desc), Full count (asc)")
    print("=" * 100)

    for key in sorted(concept_groups.keys()):
        s = concept_groups[key]
        folder_short = Path(s.folder).name
        print(f"\n{'='*80}")
        print(f"📌 {folder_short} ({s.name})")
        print(f"   Total experiments: {s.total_experiments}")
        print(f"   Best: {s.best_success_rate:.1f}% | Avg: {s.avg_success_rate:.1f}% | Worst: {s.worst_success_rate:.1f}%")
        print(f"{'='*80}")

        for i, r in enumerate(s.top_results, 1):
            params_str = ", ".join([f"{k}={v}" for k, v in r.params.items()])
            print(f"\n   🥇 Rank #{i}: {r.success_rate:.1f}% (Safe={r.safe}, Partial={r.partial}, Full={r.full})")
            print(f"      Full Path: {r.full_path}")
            print(f"      Params: {params_str}")

    # Export to CSV if requested
    if args.export:
        with open(args.export, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['concept', 'folder', 'experiments', 'best_rate', 'avg_rate', 'worst_rate', 'best_config'])
            for s in all_summaries:
                writer.writerow([
                    s.name,
                    Path(s.folder).name,
                    s.total_experiments,
                    s.best_success_rate,
                    round(s.avg_success_rate, 2),
                    s.worst_success_rate,
                    s.best_config
                ])
        print(f"\n📁 Exported to: {args.export}")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
