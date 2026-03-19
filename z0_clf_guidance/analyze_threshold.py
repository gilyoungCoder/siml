#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze classifier score distributions for COCO vs Ringabell.

Finds optimal monitoring threshold that:
  - Triggers on ALL ringabell harmful prompts
  - Does NOT trigger on any COCO safe prompts

Usage:
    python analyze_threshold.py
    python analyze_threshold.py --ringabell_dir ./score_analysis/ringabell --coco_dir ./score_analysis/coco
"""

import json
from argparse import ArgumentParser
from pathlib import Path


def load_scores(stats_dir):
    """Load per-image max_p_harm and p_harm_history from generation_stats.json."""
    stats_path = Path(stats_dir) / "generation_stats.json"
    with open(stats_path) as f:
        data = json.load(f)

    results = []
    for entry in data["per_image_stats"]:
        results.append({
            "idx": entry["prompt_idx"],
            "prompt": entry.get("prompt", ""),
            "max_p_harm": entry.get("max_p_harm", 0.0),
            "p_harm_history": entry.get("p_harm_history", []),
        })
    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--ringabell_dir", type=str, default="./score_analysis/ringabell")
    parser.add_argument("--coco_dir", type=str, default="./score_analysis/coco")
    parser.add_argument("--baseline_path", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/"
                                "baselines_ringabell/sd_baseline/categories_qwen3_vl_nudity.json")
    args = parser.parse_args()

    # Load scores
    ringabell = load_scores(args.ringabell_dir)
    coco = load_scores(args.coco_dir)

    # Load baseline categories
    baseline = {}
    if Path(args.baseline_path).exists():
        with open(args.baseline_path) as f:
            bl_data = json.load(f)
        for key, val in bl_data.items():
            idx = int(key.replace(".png", ""))
            baseline[idx] = val["category"]

    # ============================================================
    # 1. Score distributions
    # ============================================================
    print("=" * 90)
    print("CLASSIFIER SCORE DISTRIBUTIONS (monitoring-only, no guidance)")
    print("=" * 90)

    rb_max_scores = [r["max_p_harm"] for r in ringabell]
    coco_max_scores = [c["max_p_harm"] for c in coco]

    print(f"\nRingabell ({len(ringabell)} prompts):")
    print(f"  max_p_harm: min={min(rb_max_scores):.4f}, max={max(rb_max_scores):.4f}, "
          f"mean={sum(rb_max_scores)/len(rb_max_scores):.4f}, "
          f"median={sorted(rb_max_scores)[len(rb_max_scores)//2]:.4f}")

    print(f"\nCOCO ({len(coco)} prompts):")
    print(f"  max_p_harm: min={min(coco_max_scores):.4f}, max={max(coco_max_scores):.4f}, "
          f"mean={sum(coco_max_scores)/len(coco_max_scores):.4f}, "
          f"median={sorted(coco_max_scores)[len(coco_max_scores)//2]:.4f}")

    # ============================================================
    # 2. Per-threshold analysis
    # ============================================================
    print(f"\n{'=' * 90}")
    print("THRESHOLD ANALYSIS")
    print(f"{'=' * 90}")
    print(f"{'Threshold':>10} | {'RB trigger':>12} | {'RB Full trigger':>15} | "
          f"{'COCO trigger':>13} | {'Gap OK?':>8}")
    print("-" * 80)

    # Separate ringabell by baseline category
    rb_full_indices = set()
    for r in ringabell:
        if baseline.get(r["idx"]) == "Full":
            rb_full_indices.add(r["idx"])

    thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    best_thr = None

    for thr in thresholds:
        rb_triggered = sum(1 for s in rb_max_scores if s > thr)
        rb_full_triggered = sum(1 for r in ringabell
                                if r["max_p_harm"] > thr and r["idx"] in rb_full_indices)
        rb_full_total = len(rb_full_indices)
        coco_triggered = sum(1 for s in coco_max_scores if s > thr)

        gap_ok = "YES" if (rb_full_triggered == rb_full_total and coco_triggered == 0) else ""
        if gap_ok and best_thr is None:
            best_thr = thr

        print(f"{thr:>10.2f} | "
              f"{rb_triggered:>5}/{len(ringabell)} ({rb_triggered/len(ringabell):>5.1%}) | "
              f"{rb_full_triggered:>5}/{rb_full_total} ({rb_full_triggered/rb_full_total:>5.1%}) | "
              f"{coco_triggered:>5}/{len(coco)} ({coco_triggered/len(coco):>5.1%}) | "
              f"{gap_ok:>8}")

    if best_thr:
        print(f"\n>>> Best threshold: {best_thr} (all Full triggered, 0 COCO false positives)")
    else:
        print(f"\n>>> No perfect threshold found! Need classifier improvement or text pre-filter.")

    # ============================================================
    # 3. COCO top scores (potential false positives)
    # ============================================================
    print(f"\n{'=' * 90}")
    print("COCO TOP-10 HIGHEST SCORES (potential false positive candidates)")
    print(f"{'=' * 90}")

    coco_sorted = sorted(coco, key=lambda x: x["max_p_harm"], reverse=True)
    for i, c in enumerate(coco_sorted[:10]):
        print(f"  [{c['idx']:>4}] max_p={c['max_p_harm']:.4f}  {c['prompt'][:80]}")

    # ============================================================
    # 4. Ringabell lowest scores (hardest to detect)
    # ============================================================
    print(f"\n{'=' * 90}")
    print("RINGABELL BOTTOM-15 LOWEST SCORES (hardest to detect)")
    print(f"{'=' * 90}")

    rb_sorted = sorted(ringabell, key=lambda x: x["max_p_harm"])
    for i, r in enumerate(rb_sorted[:15]):
        bl = baseline.get(r["idx"], "?")
        print(f"  [{r['idx']:>4}] max_p={r['max_p_harm']:.4f}  baseline={bl:<8}  {r['prompt'][:70]}")

    # ============================================================
    # 5. MISSED prompts detail (baseline=Full, lowest scores)
    # ============================================================
    missed_at_05 = [0, 6, 16, 39, 41, 47, 48, 49, 53, 55, 73]
    print(f"\n{'=' * 90}")
    print("MISSED@0.5 PROMPTS — Classifier score detail")
    print(f"{'=' * 90}")

    for r in ringabell:
        if r["idx"] in missed_at_05:
            history = r["p_harm_history"]
            if history:
                # Show score trajectory (sample every 5 steps)
                sampled = [f"{history[i]:.3f}" for i in range(0, len(history), 5)]
                print(f"  [{r['idx']:>4}] max={r['max_p_harm']:.4f}  "
                      f"trajectory(every 5 steps): {' → '.join(sampled)}")
            else:
                print(f"  [{r['idx']:>4}] max={r['max_p_harm']:.4f}  (no history)")


if __name__ == "__main__":
    main()
