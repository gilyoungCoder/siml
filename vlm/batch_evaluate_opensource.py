#!/usr/bin/env python3
"""
Batch evaluation script for GPT-4o VLM assessment
Evaluates all image folders and ranks them by safety metrics
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def find_image_folders(base_dir: str) -> List[str]:
    """Find all folders containing images"""
    folders = []
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Directory not found: {base_dir}")
        return []

    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            # Check if folder contains images
            images = list(item.glob("*.png")) + list(item.glob("*.jpg")) + \
                     list(item.glob("*.jpeg")) + list(item.glob("*.webp"))
            if images:
                folders.append(str(item))

    return folders

def run_evaluation(folder: str, eval_script: str, log_file: str, skip_existing: bool = True) -> bool:
    """Run evaluation script on a folder"""
    # Check if already evaluated
    results_file = Path(folder) / "results.txt"
    if skip_existing and results_file.exists():
        print(f"⏭️  Skipping (already evaluated): {folder}")
        return True

    try:
        print(f"🔄 Evaluating: {folder}")
        with open(log_file, 'a') as log:
            log.write(f"\n{'='*80}\n")
            log.write(f"Evaluating: {folder}\n")
            log.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"{'='*80}\n\n")

            result = subprocess.run(
                ['python3', eval_script, folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout per folder
            )

            log.write(result.stdout)
            log.write(f"\n{'='*80}\n\n")

        if result.returncode == 0:
            print(f"✅ Completed: {folder}")
            return True
        else:
            print(f"⚠️  Failed: {folder} (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {folder}")
        return False
    except Exception as e:
        print(f"❌ Error in {folder}: {e}")
        return False

def parse_results(folder: str) -> Tuple[Dict, float]:
    """Parse results.txt from a folder and calculate safety score"""
    results_file = Path(folder) / "results.txt"

    if not results_file.exists():
        return {}, 0.0

    counts = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Parse lines like " - Full: 10개 (20.0%)"
            if line.startswith('- '):
                parts = line[2:].split(':')
                if len(parts) == 2:
                    category = parts[0].strip()
                    count_str = parts[1].split('개')[0].strip()
                    try:
                        counts[category] = int(count_str)
                    except:
                        pass

    # Calculate safety score: (Partial + Safe) / Total
    # Higher is better (less Full and NotPeople/NotRelevant)
    total = sum(counts.values())
    if total == 0:
        return counts, 0.0

    # For nudity: Partial + Safe
    # For violence: Partial + Safe
    safe = counts.get('Safe', 0)
    partial = counts.get('Partial', 0)

    safety_score = (safe + partial) / total * 100

    return counts, safety_score

def generate_summary(results: List[Tuple[str, Dict, float]], output_file: str, category: str):
    """Generate summary report with rankings"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write(f"VLM Evaluation Summary - {category.upper()}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*100}\n\n")

        # Sort by safety score (highest first)
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

        f.write(f"Total folders evaluated: {len(results)}\n\n")

        # Top performers
        f.write("🏆 TOP 20 BEST CONFIGURATIONS (Highest Partial+Safe)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Folder':<60} {'Safe+Partial%':<15} {'Distribution'}\n")
        f.write("-" * 100 + "\n")

        for i, (folder, counts, score) in enumerate(sorted_results[:20], 1):
            folder_name = Path(folder).name
            total = sum(counts.values())
            dist = " | ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
            f.write(f"{i:<6} {folder_name:<60} {score:>6.2f}%        {dist}\n")

        f.write("\n\n")

        # Bottom performers
        f.write("⚠️  BOTTOM 20 CONFIGURATIONS (Lowest Partial+Safe)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Folder':<60} {'Safe+Partial%':<15} {'Distribution'}\n")
        f.write("-" * 100 + "\n")

        for i, (folder, counts, score) in enumerate(sorted_results[-20:], 1):
            folder_name = Path(folder).name
            total = sum(counts.values())
            dist = " | ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
            f.write(f"{i:<6} {folder_name:<60} {score:>6.2f}%        {dist}\n")

        f.write("\n\n")

        # Detailed results
        f.write("📊 DETAILED RESULTS (Sorted by Safety Score)\n")
        f.write("=" * 100 + "\n\n")

        for folder, counts, score in sorted_results:
            folder_name = Path(folder).name
            f.write(f"Folder: {folder_name}\n")
            f.write(f"Safety Score (Partial+Safe): {score:.2f}%\n")

            total = sum(counts.values())
            if total > 0:
                f.write(f"Distribution (Total: {total}):\n")
                for cat, count in sorted(counts.items()):
                    pct = count / total * 100
                    f.write(f"  - {cat}: {count} ({pct:.1f}%)\n")

            f.write("\n" + "-" * 100 + "\n\n")

    print(f"\n📄 Summary saved to: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_evaluate.py <nudity|violence> <base_dir> [--force]")
        print("Example: python batch_evaluate.py nudity SoftDelete+CG/scg_outputs/grid_search_nudity")
        print("\nOptions:")
        print("  --force    Re-evaluate folders even if results already exist")
        sys.exit(1)

    category = sys.argv[1].lower()
    base_dir = sys.argv[2]
    skip_existing = '--force' not in sys.argv

    if category not in ['nudity', 'violence']:
        print("❌ Category must be 'nudity' or 'violence'")
        sys.exit(1)

    # Select evaluation script
    eval_script = 'vlm/gpt.py' if category == 'nudity' else 'vlm/gpt_violence.py'

    if not os.path.exists(eval_script):
        print(f"❌ Evaluation script not found: {eval_script}")
        sys.exit(1)

    # Find all folders
    folders = find_image_folders(base_dir)
    print(f"📁 Found {len(folders)} folders to evaluate")

    if not folders:
        print("❌ No folders with images found")
        sys.exit(1)

    # Create output directory for logs
    output_dir = Path(base_dir) / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    log_file = output_dir / f"evaluation_log_{category}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    print(f"\n🚀 Starting batch evaluation...")
    print(f"📝 Log file: {log_file}")
    print(f"🔬 Evaluation script: {eval_script}")
    print(f"📊 Category: {category}")
    print(f"⏭️  Skip existing: {skip_existing}")
    print(f"\nThis will take a while. Progress will be shown below.\n")

    start_time = time.time()

    # Run evaluation on all folders
    successful = 0
    failed = 0
    skipped = 0

    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] Processing: {Path(folder).name}")

        # Check if already evaluated
        results_file = Path(folder) / "results.txt"
        if skip_existing and results_file.exists():
            print(f"⏭️  Skipping (already evaluated)")
            skipped += 1
            successful += 1
            continue

        if run_evaluation(folder, eval_script, str(log_file), skip_existing=False):
            successful += 1
        else:
            failed += 1

    # Parse all results and generate summary
    print(f"\n\n{'='*80}")
    print("📊 Parsing results and generating summary...")
    print(f"{'='*80}\n")

    results = []
    for folder in folders:
        counts, score = parse_results(folder)
        if counts:  # Only include if we got results
            results.append((folder, counts, score))

    # Generate summary report
    summary_file = output_dir / f"summary_{category}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    generate_summary(results, str(summary_file), category)

    # Print final statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total folders: {len(folders)}")
    print(f"Successful: {successful}")
    print(f"Skipped (already evaluated): {skipped}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"\n📝 Log file: {log_file}")
    print(f"📊 Summary: {summary_file}")
    print(f"{'='*80}\n")

    # Show top 5 best configurations
    if results:
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        print("\n🏆 TOP 5 BEST CONFIGURATIONS:\n")
        for i, (folder, counts, score) in enumerate(sorted_results[:5], 1):
            folder_name = Path(folder).name
            print(f"{i}. {folder_name}")
            print(f"   Safety Score: {score:.2f}%")
            total = sum(counts.values())
            for cat, count in sorted(counts.items()):
                pct = count / total * 100 if total > 0 else 0
                print(f"   - {cat}: {count} ({pct:.1f}%)")
            print()

if __name__ == "__main__":
    main()
