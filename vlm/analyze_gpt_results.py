#!/usr/bin/env python
"""
GPT-4o VLM 평가 결과 요약 분석

Usage:
    python analyze_gpt_results.py [--export results_gpt_summary.csv]
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
    best_full: int
    best_safe: int
    best_partial: int
    top_results: List[ExperimentResult]


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

    thr_match = re.search(r'thr([\d.]+)-([\d.]+)', exp_name)
    if thr_match:
        params['thr'] = f"{thr_match.group(1)}→{thr_match.group(2)}"

    cgs_match = re.search(r'cgs([\d.]+)', exp_name)
    if cgs_match:
        params['cgs'] = float(cgs_match.group(1))

    if 'linear' in exp_name or 'lin' in exp_name:
        params['strategy'] = 'linear'
    elif 'cosine' in exp_name or 'cos' in exp_name:
        params['strategy'] = 'cosine'

    return params


def parse_result_file(filepath: Path) -> Optional[Dict]:
    """Parse results_gpt4o_*.txt file."""
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

    result_filename = f"results_gpt4o_{concept}.txt"
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


def analyze_baseline(img_dir: str, concept: str) -> Optional[Dict]:
    """Analyze baseline result (single folder, not grid search)."""
    base_path = Path(img_dir)
    if not base_path.exists():
        return None

    result_file = base_path / f"results_gpt4o_{concept}.txt"
    if not result_file.exists():
        return None

    parsed = parse_result_file(result_file)
    if parsed:
        return {
            'name': base_path.name,
            'full_path': str(base_path),
            'success_rate': parsed['success_rate'],
            'full': parsed['full'],
            'safe': parsed['safe'],
            'partial': parsed['partial']
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze all GPT-4o evaluation results")
    parser.add_argument('--export', type=str, default=None, help='Export to CSV')
    parser.add_argument('--top', type=int, default=3, help='Number of top results to show per folder')
    args = parser.parse_args()

    base_unlearning = "/mnt/home/yhgil99/unlearning"

    # Define baselines
    baselines = {
        "nudity": [
            ("SAFREE/safree_outputs/i2p/sexual", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/sexual", "SD 1.4 Baseline"),
        ],
        "violence": [
            ("SAFREE/safree_outputs/i2p/violence", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/violence", "SD 1.4 Baseline"),
        ],
        "harassment": [
            ("SAFREE/safree_outputs/i2p/harassment", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/harassment", "SD 1.4 Baseline"),
        ],
        "hate": [
            ("SAFREE/safree_outputs/i2p/hate", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/hate", "SD 1.4 Baseline"),
        ],
        "shocking": [
            ("SAFREE/safree_outputs/i2p/shocking", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/shocking", "SD 1.4 Baseline"),
        ],
        "illegal": [
            ("SAFREE/safree_outputs/i2p/illegal activity", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/illegal_activity", "SD 1.4 Baseline"),
        ],
        "self_harm": [
            ("SAFREE/safree_outputs/i2p/self-harm", "SAFREE Baseline"),
            ("SoftDelete+CG/SDbaseline/i2p/selfharm", "SD 1.4 Baseline"),
        ],
    }

    # Define grid search folders to analyze
    folders_to_analyze = [
        # Nudity
        ("SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class", "nudity"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always", "nudity"),
        ("SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700", "nudity"),
        ("SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip", "nudity"),
        ("SAFREE/results/grid_search_safree_4class_nudity", "nudity"),

        # Violence
        ("SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500", "violence"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip", "violence"),
        ("SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400", "violence"),
        ("SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip", "violence"),
        ("SAFREE/results/grid_search_safree_9class/violence_step15500", "violence"),

        # Harassment
        ("SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300", "harassment"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip", "harassment"),
        ("SAFREE/results/grid_search_safree_9class/harassment_step24300", "harassment"),

        # Hate
        ("SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800", "hate"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip", "hate"),
        ("SAFREE/results/grid_search_safree_9class/hate_step20800", "hate"),

        # Shocking
        ("SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700", "shocking"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip", "shocking"),
        ("SAFREE/results/grid_search_safree_9class/shocking_step23700", "shocking"),

        # Illegal
        ("SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600", "illegal"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip", "illegal"),
        ("SAFREE/results/grid_search_safree_9class/illegal_step22600", "illegal"),

        # Self-harm
        ("SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700", "self_harm"),
        ("SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip", "self_harm"),
        ("SAFREE/results/grid_search_safree_9class/selfharm_step20700", "self_harm"),
    ]

    print("\n" + "=" * 100)
    print("📊 GPT-4o VLM EVALUATION RESULTS SUMMARY")
    print("=" * 100)

    # Analyze baselines first
    print("\n" + "=" * 100)
    print("📌 BASELINE RESULTS")
    print("=" * 100)
    print(f"\n{'Concept':<15} {'Method':<20} {'Success%':>10} {'Safe':>6} {'Partial':>8} {'Full':>6}")
    print("-" * 70)

    baseline_results = {}
    for concept, bl_list in baselines.items():
        baseline_results[concept] = []
        for rel_path, name in bl_list:
            full_path = os.path.join(base_unlearning, rel_path)
            result = analyze_baseline(full_path, concept)
            if result:
                baseline_results[concept].append(result)
                print(f"{concept:<15} {name:<20} {result['success_rate']:>9.1f}% {result['safe']:>6} {result['partial']:>8} {result['full']:>6}")

    # Analyze grid search folders
    all_summaries = []
    for rel_path, concept in folders_to_analyze:
        full_path = os.path.join(base_unlearning, rel_path)
        summary = analyze_folder(full_path, concept, top_n=args.top)
        if summary and summary.total_experiments > 0:
            all_summaries.append(summary)

    # Group by concept
    concept_groups = {}
    for s in all_summaries:
        folder_name = Path(s.folder).name
        key = f"{s.name} ({folder_name})"
        concept_groups[key] = s

    # Print grid search results table
    print("\n" + "=" * 100)
    print("📊 GRID SEARCH RESULTS SUMMARY")
    print("   (Success Rate = Safe + Partial, higher is better)")
    print("=" * 100)
    print(f"\n{'Folder':<60} {'Exp':>5} {'Best':>8} {'Avg':>8} {'Worst':>8}")
    print("-" * 100)

    for key in sorted(concept_groups.keys()):
        s = concept_groups[key]
        folder_short = Path(s.folder).name[:55]
        print(f"{folder_short:<60} {s.total_experiments:>5} {s.best_success_rate:>7.1f}% {s.avg_success_rate:>7.1f}% {s.worst_success_rate:>7.1f}%")

    # Print top configs with full paths
    print("\n" + "=" * 100)
    print(f"🏆 TOP {args.top} CONFIGURATIONS BY FOLDER (Highest Success Rate)")
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
            print(f"      Path: {r.full_path}")
            print(f"      Params: {params_str}")

    # Find overall best per concept
    print("\n" + "=" * 100)
    print("🏆 BEST CONFIG PER CONCEPT (across all folders)")
    print("=" * 100)

    concept_best = {}
    for s in all_summaries:
        concept = s.name
        if s.top_results:
            best = s.top_results[0]
            if concept not in concept_best or best.success_rate > concept_best[concept].success_rate:
                concept_best[concept] = best

    for concept in sorted(concept_best.keys()):
        best = concept_best[concept]
        print(f"\n📌 {concept.upper()}")
        print(f"   Success Rate: {best.success_rate:.1f}% (Safe={best.safe}, Partial={best.partial}, Full={best.full})")
        print(f"   Path: {best.full_path}")
        params_str = ", ".join([f"{k}={v}" for k, v in best.params.items()])
        print(f"   Params: {params_str}")

    # Export to CSV if requested
    if args.export:
        with open(args.export, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['concept', 'folder', 'config', 'success_rate', 'safe', 'partial', 'full', 'full_path'])

            # Write baselines
            for concept, bl_results in baseline_results.items():
                for r in bl_results:
                    writer.writerow([
                        concept, 'baseline', r['name'],
                        r['success_rate'], r['safe'], r['partial'], r['full'],
                        r['full_path']
                    ])

            # Write grid search results
            for s in all_summaries:
                for r in s.top_results:
                    writer.writerow([
                        s.name, Path(s.folder).name, r.exp_name,
                        r.success_rate, r.safe, r.partial, r.full,
                        r.full_path
                    ])
        print(f"\n📁 Exported to: {args.export}")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
