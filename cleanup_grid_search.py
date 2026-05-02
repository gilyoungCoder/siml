#!/usr/bin/env python3
"""
Clean up grid search image outputs, keeping only top-10 configs per (version, dataset).
Deletes PNG images (keeps JSON/CSV/log metadata).

Usage:
    python cleanup_grid_search.py --dry-run   # Preview what will be deleted
    python cleanup_grid_search.py              # Actually delete
"""

import csv
import json
import os
import glob
import shutil
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser


def compute_metrics_from_categories(json_path):
    """Compute sr and sr_full from categories JSON."""
    with open(json_path) as f:
        data = json.load(f)
    total = len(data)
    if total == 0:
        return {"sr": 0, "sr_full": 0, "total": 0}
    counts = {"NotRel": 0, "Safe": 0, "Partial": 0, "Full": 0}
    for entry in data.values():
        cat = entry.get("category", "Unknown")
        if cat in counts:
            counts[cat] += 1
    sr = (counts["Safe"] + counts["Partial"]) / total
    sr_full = 1 - counts["Full"] / total
    return {"sr": sr, "sr_full": sr_full, "total": total}


def get_dir_image_size(dir_path):
    """Get total size of PNG files in a directory (in bytes)."""
    total = 0
    for f in os.scandir(dir_path):
        if f.name.endswith('.png') and f.is_file():
            total += f.stat().st_size
    return total


def delete_images_in_dir(dir_path, dry_run=True):
    """Delete all PNG files in a directory. Returns (count, bytes)."""
    count = 0
    total_bytes = 0
    for f in os.scandir(dir_path):
        if f.name.endswith('.png') and f.is_file():
            size = f.stat().st_size
            total_bytes += size
            count += 1
            if not dry_run:
                os.remove(f.path)
    return count, total_bytes


def human_size(nbytes):
    """Convert bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"


def process_csv_tracked(csv_path, version_to_dir, base_dir, dry_run=True):
    """
    Process directories tracked by a results CSV.
    Returns (total_deleted_bytes, total_deleted_files).
    """
    # Read CSV and group by (version, dataset)
    groups = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['version'], row['dataset'])
            groups[key].append(row)

    total_deleted_bytes = 0
    total_deleted_files = 0
    total_kept = 0

    for (version, dataset), rows in sorted(groups.items()):
        # Sort by sr descending, then sr_full descending
        rows.sort(key=lambda r: (float(r.get('sr', 0)), float(r.get('sr_full', 0))), reverse=True)

        top10_names = set(r['exp_name'] for r in rows[:10])
        rest = [r for r in rows[10:]]

        # Get directory base
        if version not in version_to_dir:
            print(f"  [SKIP] Unknown version: {version}")
            continue

        dir_base = os.path.join(base_dir, version_to_dir[version], dataset)
        if not os.path.isdir(dir_base):
            print(f"  [SKIP] Dir not found: {dir_base}")
            continue

        group_deleted_bytes = 0
        group_deleted_files = 0

        for row in rest:
            exp_dir = os.path.join(dir_base, row['exp_name'])
            if os.path.isdir(exp_dir):
                count, nbytes = delete_images_in_dir(exp_dir, dry_run=dry_run)
                group_deleted_files += count
                group_deleted_bytes += nbytes

        total_deleted_bytes += group_deleted_bytes
        total_deleted_files += group_deleted_files
        total_kept += len(top10_names)

        # Show top 3 kept configs
        top3 = rows[:3]
        top3_info = ", ".join(f"sr={r['sr']}" for r in top3)
        print(f"  ({version}, {dataset}): keep {len(top10_names)}, delete {len(rest)} configs "
              f"({group_deleted_files} images, {human_size(group_deleted_bytes)})")
        print(f"    Top 3: {top3_info}")

    return total_deleted_bytes, total_deleted_files


def process_untracked_dir(grid_dir, dry_run=True):
    """
    Process a grid search directory not tracked by any CSV.
    Scans for categories JSON, ranks, keeps top 10.
    Returns (deleted_bytes, deleted_files).
    """
    if not os.path.isdir(grid_dir):
        return 0, 0

    # Check if it has dataset subdirs (ringabell, etc.) or direct config subdirs
    subdirs = [d for d in os.scandir(grid_dir) if d.is_dir() and d.name != 'logs']
    if not subdirs:
        return 0, 0

    # Check if first subdir contains images directly or has further subdirs
    first_sub = subdirs[0]
    has_pngs = any(f.name.endswith('.png') for f in os.scandir(first_sub.path) if f.is_file())
    has_categories = any(f.name.startswith('categories_') for f in os.scandir(first_sub.path) if f.is_file())

    # Check if this is a dataset-level dir (contains ringabell/, mma/, etc.)
    sub_names = set(d.name for d in subdirs)
    dataset_names = {'ringabell', 'mma', 'unlearndiff', 'coco', 'p4dn', 'i2p'}
    is_dataset_level = bool(sub_names & dataset_names)

    total_deleted_bytes = 0
    total_deleted_files = 0

    if is_dataset_level:
        # Process each dataset subdir
        for ds_dir in subdirs:
            if ds_dir.name in dataset_names:
                db, df = _process_config_dirs(ds_dir.path, f"{os.path.basename(grid_dir)}/{ds_dir.name}", dry_run)
                total_deleted_bytes += db
                total_deleted_files += df
    else:
        # Direct config subdirs
        db, df = _process_config_dirs(grid_dir, os.path.basename(grid_dir), dry_run)
        total_deleted_bytes += db
        total_deleted_files += df

    return total_deleted_bytes, total_deleted_files


def _process_config_dirs(parent_dir, label, dry_run):
    """Process config subdirs within a parent dir. Rank by sr, keep top 10."""
    configs = []
    no_eval_dirs = []

    for d in os.scandir(parent_dir):
        if not d.is_dir() or d.name == 'logs':
            continue

        # Look for categories JSON
        cat_files = [f for f in os.scandir(d.path) if f.is_file() and f.name.startswith('categories_')]
        if cat_files:
            try:
                metrics = compute_metrics_from_categories(cat_files[0].path)
                configs.append({
                    'name': d.name,
                    'path': d.path,
                    'sr': metrics['sr'],
                    'sr_full': metrics['sr_full'],
                    'total': metrics['total'],
                })
            except:
                no_eval_dirs.append(d.path)
        else:
            no_eval_dirs.append(d.path)

    if not configs and not no_eval_dirs:
        return 0, 0

    # Sort by sr desc, sr_full desc
    configs.sort(key=lambda c: (c['sr'], c['sr_full']), reverse=True)

    top10 = configs[:10]
    to_delete = configs[10:]

    deleted_bytes = 0
    deleted_files = 0

    for cfg in to_delete:
        count, nbytes = delete_images_in_dir(cfg['path'], dry_run=dry_run)
        deleted_files += count
        deleted_bytes += nbytes

    # Also delete images in dirs with no eval (they're useless without results)
    for d_path in no_eval_dirs:
        count, nbytes = delete_images_in_dir(d_path, dry_run=dry_run)
        deleted_files += count
        deleted_bytes += nbytes

    top_sr = f"sr={configs[0]['sr']:.4f}" if configs else "N/A"
    print(f"  {label}: keep {len(top10)}, delete {len(to_delete)} evaluated + {len(no_eval_dirs)} unevaluated "
          f"({deleted_files} images, {human_size(deleted_bytes)})")
    if configs:
        print(f"    Best: {top_sr}, Worst kept: sr={configs[min(9,len(configs)-1)]['sr']:.4f}")

    return deleted_bytes, deleted_files


def main():
    parser = ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Preview what would be deleted without actually deleting')
    args = parser.parse_args()

    dry_run = args.dry_run
    mode_str = "DRY RUN" if dry_run else "DELETING"
    print(f"{'='*60}")
    print(f"Grid Search Cleanup ({mode_str})")
    print(f"{'='*60}\n")

    grand_total_bytes = 0
    grand_total_files = 0

    # ──────────────────────────────────────────────────
    # 1. SoftDelete+CG - CSV-tracked
    # ──────────────────────────────────────────────────
    scg_base = "/mnt/home/yhgil99/unlearning/SoftDelete+CG"
    scg_csv = os.path.join(scg_base, "scg_all_results.csv")

    if os.path.exists(scg_csv):
        print("── SoftDelete+CG (CSV-tracked) ──")
        scg_version_map = {
            "unified_v1": "scg_outputs/unified_grid",
            "unified_v2": "scg_outputs/unified_grid_v2",
        }
        db, df = process_csv_tracked(scg_csv, scg_version_map, scg_base, dry_run)
        grand_total_bytes += db
        grand_total_files += df
        print()

    # ──────────────────────────────────────────────────
    # 2. SoftDelete+CG - Old grid search dirs (not in CSV)
    # ──────────────────────────────────────────────────
    print("── SoftDelete+CG (older grid search dirs) ──")
    scg_old_grid_dirs = [
        "grid_search_adaptive_spatial_cg_nudity",
        "grid_search_adaptive_spatial_cg_vangogh",
        "grid_search_adaptive_spatial_cg_violence",
        "grid_search_dual_nudity_20260127_225621",
        "grid_search_dual_ringabell_20260128_201546",
        "grid_search_sexual_20260127_190422",
        "grid_search_ringabell_20260128_201546",
        "mon01_grid",
        "fine_grid_mon4class",
        "grid_search_nudity_6class_v2_step22700",
        "grid_search_nudity_6class_v2_step22700_skip",
        "grid_search_violence_13class_step28400",
        "grid_search_violence_13class_step28400_skip",
        "grid_search_jtt_step18400",
        "grid_mon4class_ringabell_20260130_193851",
        "grid_mon4class_ringabell_20260130_195313",
        "grid_mon4class_unlearndiff_20260130_194724",
        "grid_mon4class_unlearndiff_20260130_200205",
        "grid_mon4class_p4dn_20260130_194316",
        "grid_mon4class_p4dn_20260130_195737",
        "grid_safree_mon_p4dn_20260130_194310",
        "grid_safree_mon_p4dn_20260130_195702",
    ]
    for dirname in scg_old_grid_dirs:
        full_path = os.path.join(scg_base, "scg_outputs", dirname)
        if os.path.isdir(full_path):
            db, df = process_untracked_dir(full_path, dry_run)
            grand_total_bytes += db
            grand_total_files += df

    # grid_search_results has deeper structure: category/config/images
    print("\n── SoftDelete+CG (grid_search_results - per category) ──")
    gsr_path = os.path.join(scg_base, "scg_outputs", "grid_search_results")
    if os.path.isdir(gsr_path):
        for cat_dir in sorted(os.scandir(gsr_path), key=lambda x: x.name):
            if cat_dir.is_dir():
                db, df = _process_config_dirs(cat_dir.path, f"grid_search_results/{cat_dir.name}", dry_run)
                grand_total_bytes += db
                grand_total_files += df
    print()

    # ──────────────────────────────────────────────────
    # 3. z0_clf_guidance - CSV-tracked
    # ──────────────────────────────────────────────────
    z0_base = "/mnt/home/yhgil99/unlearning/z0_clf_guidance"
    z0_csv = os.path.join(z0_base, "z0_all_results.csv")

    if os.path.exists(z0_csv):
        print("── z0_clf_guidance (CSV-tracked) ──")
        z0_version_map = {
            "monitoring": "grid_monitoring_output",
            "v2": "grid_v2_output",
            "v3": "grid_v3_output",
            "v4": "grid_v4_output",
            "v5": "grid_v5_output",
            "v5b": "grid_v5b_output",
        }
        db, df = process_csv_tracked(z0_csv, z0_version_map, z0_base, dry_run)
        grand_total_bytes += db
        grand_total_files += df
        print()

    # ──────────────────────────────────────────────────
    # 4. z0_clf_guidance - Old grid_search_output (not in CSV)
    # ──────────────────────────────────────────────────
    z0_old = os.path.join(z0_base, "grid_search_output")
    if os.path.isdir(z0_old):
        print("── z0_clf_guidance (old grid_search_output) ──")
        # This has a deeper structure: grid_search_output/grid_YYYYMMDD_HHMMSS/config_name/
        for sub in sorted(os.scandir(z0_old), key=lambda x: x.name):
            if sub.is_dir() and sub.name.startswith('grid_'):
                db, df = process_untracked_dir(sub.path, dry_run)
                grand_total_bytes += db
                grand_total_files += df
        print()

    # ──────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"TOTAL: {grand_total_files:,} images, {human_size(grand_total_bytes)}")
    if dry_run:
        print("(DRY RUN - nothing was deleted)")
        print(f"\nRun without --dry-run to actually delete.")
    else:
        print("(DELETED)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
