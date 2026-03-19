#!/usr/bin/env python
"""
Best config들을 자동으로 찾아서 이미지 그리드로 비교

GPT-4o 평가 완료 후 concept별 best config를 찾아서 시각적으로 비교합니다.

Usage:
    # 특정 concept의 best configs 비교
    python compare_best_configs.py --concept nudity --output grids/nudity_comparison.png

    # 모든 concept 비교 그리드 생성
    python compare_best_configs.py --all --output-dir grids/

    # Baseline과 best config 비교
    python compare_best_configs.py --concept violence --with-baseline --output grids/violence_vs_baseline.png
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL not installed. Run: pip install Pillow")
    exit(1)

from make_grid import get_images_from_folder, sample_images, create_comparison_grid


BASE_DIR = "/mnt/home/yhgil99/unlearning"


# Baseline directories
BASELINES = {
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

# Grid search folders per concept
GRID_SEARCH_FOLDERS = {
    "nudity": [
        "SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class",
        "SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always",
        "SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700",
        "SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip",
        "SAFREE/results/grid_search_safree_4class_nudity",
    ],
    "violence": [
        "SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500",
        "SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip",
        "SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400",
        "SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip",
        "SAFREE/results/grid_search_safree_9class/violence_step15500",
    ],
    "harassment": [
        "SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300",
        "SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip",
        "SAFREE/results/grid_search_safree_9class/harassment_step24300",
    ],
    "hate": [
        "SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800",
        "SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip",
        "SAFREE/results/grid_search_safree_9class/hate_step20800",
    ],
    "shocking": [
        "SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700",
        "SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip",
        "SAFREE/results/grid_search_safree_9class/shocking_step23700",
    ],
    "illegal": [
        "SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600",
        "SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip",
        "SAFREE/results/grid_search_safree_9class/illegal_step22600",
    ],
    "self_harm": [
        "SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700",
        "SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip",
        "SAFREE/results/grid_search_safree_9class/selfharm_step20700",
    ],
}


def parse_result_file(filepath: Path) -> Optional[Dict]:
    """Parse results_gpt4o_*.txt file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        total_match = re.search(r'Total images: (\d+)', content)
        total = int(total_match.group(1)) if total_match else 0

        dist = {}
        dist_pattern = r'- (\w+): (\d+) \(([\d.]+)%\)'
        for match in re.finditer(dist_pattern, content):
            dist[match.group(1)] = int(match.group(2))

        safe = dist.get('Safe', 0)
        partial = dist.get('Partial', 0)
        full = dist.get('Full', 0)
        success_rate = (safe + partial) / total * 100 if total > 0 else 0.0

        return {
            'success_rate': success_rate,
            'full': full,
            'safe': safe,
            'partial': partial,
            'total': total
        }
    except:
        return None


def find_best_config(grid_search_folder: str, concept: str) -> Optional[Tuple[str, Dict]]:
    """Find the best config in a grid search folder based on GPT-4o results."""
    base_path = Path(grid_search_folder)
    if not base_path.exists():
        return None

    result_filename = f"results_gpt4o_{concept}.txt"
    best_config = None
    best_rate = -1

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        result_file = exp_dir / result_filename
        if not result_file.exists():
            continue

        parsed = parse_result_file(result_file)
        if parsed and parsed['success_rate'] > best_rate:
            best_rate = parsed['success_rate']
            best_config = (str(exp_dir), parsed)

    return best_config


def find_all_best_configs(concept: str, top_n: int = 3) -> List[Tuple[str, str, Dict]]:
    """Find best configs across all grid search folders for a concept."""
    results = []

    for folder_rel in GRID_SEARCH_FOLDERS.get(concept, []):
        folder_full = os.path.join(BASE_DIR, folder_rel)
        folder_path = Path(folder_full)

        if not folder_path.exists():
            continue

        result_filename = f"results_gpt4o_{concept}.txt"

        for exp_dir in folder_path.iterdir():
            if not exp_dir.is_dir():
                continue

            result_file = exp_dir / result_filename
            if not result_file.exists():
                continue

            parsed = parse_result_file(result_file)
            if parsed:
                results.append((
                    str(exp_dir),
                    f"{folder_path.name}/{exp_dir.name}",
                    parsed
                ))

    # Sort by success rate (desc), then by full count (asc)
    results.sort(key=lambda x: (x[2]['success_rate'], -x[2]['full']), reverse=True)
    return results[:top_n]


def create_concept_comparison(
    concept: str,
    output_path: str,
    with_baseline: bool = True,
    top_n: int = 3,
    images_per_row: int = 4,
    cell_size: int = 256,
    seed: int = 42
):
    """Create a comparison grid for a concept."""
    folders = []
    names = []

    # Add baselines if requested
    if with_baseline:
        for rel_path, name in BASELINES.get(concept, []):
            full_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(full_path):
                folders.append(full_path)
                names.append(name)

    # Find best configs
    best_configs = find_all_best_configs(concept, top_n=top_n)

    for full_path, short_name, stats in best_configs:
        if os.path.exists(full_path):
            folders.append(full_path)
            rate = stats['success_rate']
            names.append(f"{short_name[:30]} ({rate:.1f}%)")

    if not folders:
        print(f"❌ No valid folders found for concept: {concept}")
        return False

    # Create comparison grid
    title = f"{concept.upper()} - Best Configs Comparison (GPT-4o)"

    grid = create_comparison_grid(
        folders=folders,
        folder_names=names,
        images_per_folder=images_per_row,
        cell_size=(cell_size, cell_size),
        seed=seed,
        title=title
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    grid.save(output_path, quality=95)
    print(f"✅ Saved: {output_path}")
    print(f"   Folders compared: {len(folders)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare best configs visually")
    parser.add_argument('--concept', type=str, help='Concept to compare')
    parser.add_argument('--all', action='store_true', help='Generate grids for all concepts')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file (for single concept)')
    parser.add_argument('--output-dir', type=str, default='grids', help='Output directory (for --all)')
    parser.add_argument('--with-baseline', action='store_true', default=True, help='Include baselines')
    parser.add_argument('--no-baseline', action='store_true', help='Exclude baselines')
    parser.add_argument('--top', type=int, default=3, help='Number of top configs to show')
    parser.add_argument('--images-per-row', type=int, default=4, help='Images per row')
    parser.add_argument('--cell-size', type=int, default=256, help='Cell size in pixels')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    with_baseline = not args.no_baseline

    if args.all:
        # Generate for all concepts
        concepts = list(GRID_SEARCH_FOLDERS.keys())
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating comparison grids for {len(concepts)} concepts...")
        print(f"Output directory: {output_dir}")
        print()

        for concept in concepts:
            output_path = output_dir / f"{concept}_comparison.png"
            create_concept_comparison(
                concept=concept,
                output_path=str(output_path),
                with_baseline=with_baseline,
                top_n=args.top,
                images_per_row=args.images_per_row,
                cell_size=args.cell_size,
                seed=args.seed
            )

        print(f"\n✅ All grids saved to: {output_dir}/")

    elif args.concept:
        # Single concept
        output_path = args.output or f"{args.concept}_comparison.png"
        create_concept_comparison(
            concept=args.concept,
            output_path=output_path,
            with_baseline=with_baseline,
            top_n=args.top,
            images_per_row=args.images_per_row,
            cell_size=args.cell_size,
            seed=args.seed
        )

    else:
        print("Error: Please specify --concept or --all")
        parser.print_help()


if __name__ == '__main__':
    main()
