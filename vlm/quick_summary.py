#!/usr/bin/env python3
"""
Quick summary generator - Analyzes existing results without re-running evaluations
Useful for re-analyzing results or checking progress during batch evaluation
"""
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import time

def parse_results(folder: str) -> Tuple[Dict, float]:
    """Parse results.txt from a folder and calculate safety score"""
    results_file = Path(folder) / "results.txt"

    if not results_file.exists():
        return {}, -1.0  # -1 indicates not yet evaluated

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
    total = sum(counts.values())
    if total == 0:
        return counts, 0.0

    safe = counts.get('Safe', 0)
    partial = counts.get('Partial', 0)
    safety_score = (safe + partial) / total * 100

    return counts, safety_score

def analyze_directory(base_dir: str, category: str):
    """Analyze all results in a directory"""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ Directory not found: {base_dir}")
        return

    print(f"\n🔍 Analyzing results in: {base_dir}")
    print(f"📊 Category: {category}")
    print(f"{'='*100}\n")

    # Find all subdirectories
    folders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    results = []
    evaluated = 0
    pending = 0

    for folder in folders:
        counts, score = parse_results(str(folder))

        if score >= 0:  # Has results
            evaluated += 1
            results.append((str(folder), counts, score))
        else:
            pending += 1

    print(f"📈 Progress: {evaluated}/{len(folders)} folders evaluated ({evaluated/len(folders)*100:.1f}%)")
    print(f"⏳ Pending: {pending} folders\n")

    if not results:
        print("⚠️  No results found yet. Run evaluation first.")
        return

    # Sort by safety score
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    # Show top 10
    print("🏆 TOP 10 CONFIGURATIONS (Highest Partial+Safe):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Configuration':<60} {'Score':<10} {'Distribution'}")
    print("-" * 100)

    for i, (folder, counts, score) in enumerate(sorted_results[:10], 1):
        folder_name = Path(folder).name
        total = sum(counts.values())
        dist = " | ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
        print(f"{i:<6} {folder_name:<60} {score:>6.2f}%    {dist}")

    print("\n")

    # Show bottom 5
    print("⚠️  BOTTOM 5 CONFIGURATIONS (Lowest Partial+Safe):")
    print("-" * 100)

    for i, (folder, counts, score) in enumerate(sorted_results[-5:], 1):
        folder_name = Path(folder).name
        total = sum(counts.values())
        dist = " | ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
        print(f"{i:<6} {folder_name:<60} {score:>6.2f}%    {dist}")

    print("\n" + "=" * 100)

    # Statistics
    if len(sorted_results) > 0:
        scores = [s[2] for s in sorted_results]
        avg_score = sum(scores) / len(scores)
        print(f"\n📊 Statistics:")
        print(f"   Best Score:    {max(scores):.2f}%")
        print(f"   Worst Score:   {min(scores):.2f}%")
        print(f"   Average Score: {avg_score:.2f}%")
        print(f"   Median Score:  {sorted(scores)[len(scores)//2]:.2f}%")

    # Save quick summary
    output_dir = base_path / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    summary_file = output_dir / f"quick_summary_{category}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Quick Summary - {category.upper()}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"Progress: {evaluated}/{len(folders)} folders evaluated\n\n")

        f.write("TOP 20 CONFIGURATIONS:\n")
        f.write("-" * 100 + "\n")
        for i, (folder, counts, score) in enumerate(sorted_results[:20], 1):
            folder_name = Path(folder).name
            dist = " | ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
            f.write(f"{i}. {folder_name} - Score: {score:.2f}%\n")
            f.write(f"   {dist}\n\n")

    print(f"\n💾 Quick summary saved to: {summary_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python quick_summary.py <nudity|violence> <base_dir>")
        print("\nExamples:")
        print("  python vlm/quick_summary.py nudity SoftDelete+CG/scg_outputs/grid_search_nudity")
        print("  python vlm/quick_summary.py violence SoftDelete+CG/scg_outputs/grid_search_violence")
        sys.exit(1)

    category = sys.argv[1].lower()
    base_dir = sys.argv[2]

    if category not in ['nudity', 'violence']:
        print("❌ Category must be 'nudity' or 'violence'")
        sys.exit(1)

    analyze_directory(base_dir, category)

if __name__ == "__main__":
    main()
