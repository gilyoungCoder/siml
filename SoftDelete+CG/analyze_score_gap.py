#!/usr/bin/env python
"""
Analyze P(harm) score gap between ringabell (harmful) and COCO (safe) datasets.
Find optimal monitoring_start_step and threshold for SoftDelete+CG.

Usage:
    python analyze_score_gap.py
"""
import json
import numpy as np
from pathlib import Path

def load_step_history(stats_path):
    """Load generation_stats.json and extract per-sample, per-step P(harm)."""
    with open(stats_path) as f:
        data = json.load(f)

    samples = data["per_image_stats"]
    results = []
    for s in samples:
        if "step_history" not in s:
            continue
        steps_data = {}
        for step_info in s["step_history"]:
            step = step_info["step"]
            # P(harm) = max of class 2 (nude) and class 3 (color)
            p_harm_dict = step_info.get("p_harm", {})
            p_c2 = p_harm_dict.get("2", p_harm_dict.get(2, 0.0))
            p_c3 = p_harm_dict.get("3", p_harm_dict.get(3, 0.0))
            p_harm_max = max(p_c2, p_c3)
            p_harm_sum = p_c2 + p_c3
            steps_data[step] = {
                "p_c2": p_c2, "p_c3": p_c3,
                "p_harm_max": p_harm_max, "p_harm_sum": p_harm_sum
            }
        results.append({
            "prompt_idx": s["prompt_idx"],
            "steps": steps_data
        })
    return results


def analyze_single_step(rb_data, coco_data, step, thresholds, metric="p_harm_max"):
    """For a given step, compute TP/FP at various thresholds."""
    results = []
    for thr in thresholds:
        # TP: ringabell samples where P(harm) > threshold
        tp_count = sum(1 for s in rb_data if step in s["steps"] and s["steps"][step][metric] > thr)
        tp_rate = tp_count / len(rb_data) * 100 if rb_data else 0

        # FP: COCO samples where P(harm) > threshold
        fp_count = sum(1 for s in coco_data if step in s["steps"] and s["steps"][step][metric] > thr)
        fp_rate = fp_count / len(coco_data) * 100 if coco_data else 0

        gap = tp_rate - fp_rate
        results.append({
            "step": step, "threshold": thr,
            "tp_rate": tp_rate, "fp_rate": fp_rate, "gap": gap,
            "tp_count": tp_count, "fp_count": fp_count
        })
    return results


def analyze_sticky(rb_data, coco_data, start_step, end_step, thresholds, metric="p_harm_max"):
    """Sticky trigger: if P(harm) > thr at ANY step in [start_step, end_step] → triggered."""
    results = []
    for thr in thresholds:
        tp_count = 0
        for s in rb_data:
            triggered = any(
                step in s["steps"] and s["steps"][step][metric] > thr
                for step in range(start_step, end_step + 1)
            )
            if triggered:
                tp_count += 1
        tp_rate = tp_count / len(rb_data) * 100 if rb_data else 0

        fp_count = 0
        for s in coco_data:
            triggered = any(
                step in s["steps"] and s["steps"][step][metric] > thr
                for step in range(start_step, end_step + 1)
            )
            if triggered:
                fp_count += 1
        fp_rate = fp_count / len(coco_data) * 100 if coco_data else 0

        gap = tp_rate - fp_rate
        results.append({
            "start_step": start_step, "end_step": end_step, "threshold": thr,
            "tp_rate": tp_rate, "fp_rate": fp_rate, "gap": gap,
            "tp_count": tp_count, "fp_count": fp_count
        })
    return results


def main():
    base = Path("/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/score_analysis_4class")
    rb_data = load_step_history(base / "ringabell" / "generation_stats.json")
    coco_data = load_step_history(base / "coco" / "generation_stats.json")

    print(f"Loaded: Ringabell={len(rb_data)} samples, COCO={len(coco_data)} samples")

    # Get all available steps
    all_steps = set()
    for s in rb_data + coco_data:
        all_steps.update(s["steps"].keys())
    all_steps = sorted(all_steps)
    print(f"Steps: {min(all_steps)}-{max(all_steps)} ({len(all_steps)} total)")

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14,
                  0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ============================================
    # 1. Per-step single threshold analysis
    # ============================================
    print("\n" + "=" * 80)
    print("1. SINGLE-STEP THRESHOLD ANALYSIS (P(harm) = max(P_c2, P_c3))")
    print("=" * 80)

    best_per_step = {}
    for step in all_steps:
        results = analyze_single_step(rb_data, coco_data, step, thresholds)
        # Best gap
        best = max(results, key=lambda r: r["gap"])
        best_per_step[step] = best

    # Print summary table
    print(f"\n{'Step':>4} | {'Best Thr':>8} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 50)
    for step in all_steps:
        b = best_per_step[step]
        print(f"{step:4d} | {b['threshold']:8.2f} | {b['tp_rate']:6.1f}% | {b['fp_rate']:7.1f}% | {b['gap']:+6.1f}%")

    # Top 10 steps by gap
    sorted_steps = sorted(best_per_step.items(), key=lambda x: x[1]["gap"], reverse=True)
    print(f"\nTop 10 steps by gap:")
    for step, b in sorted_steps[:10]:
        print(f"  Step {step:2d}: thr={b['threshold']:.2f} → RB {b['tp_rate']:.1f}%, COCO {b['fp_rate']:.1f}%, gap={b['gap']:+.1f}%")

    # ============================================
    # 2. Per-step detailed (with FP tolerance)
    # ============================================
    print("\n" + "=" * 80)
    print("2. BEST THRESHOLDS WITH FP TOLERANCE (COCO FP ≤ 10%)")
    print("=" * 80)

    print(f"\n{'Step':>4} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 50)
    for step in all_steps:
        results = analyze_single_step(rb_data, coco_data, step, thresholds)
        # Filter to FP ≤ 10%, then maximize TP
        valid = [r for r in results if r["fp_rate"] <= 10.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            print(f"{step:4d} | {best['threshold']:6.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")
        else:
            print(f"{step:4d} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6}")

    # ============================================
    # 3. Sticky trigger analysis
    # ============================================
    print("\n" + "=" * 80)
    print("3. STICKY TRIGGER ANALYSIS (start_step..49, any step triggers)")
    print("=" * 80)

    # Try different start steps
    start_steps_to_try = [0, 3, 5, 7, 10, 13, 15, 20, 25, 30]

    print(f"\n{'Start':>5} | {'Best Thr':>8} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 55)

    sticky_best = {}
    for ss in start_steps_to_try:
        results = analyze_sticky(rb_data, coco_data, ss, max(all_steps), thresholds)
        best = max(results, key=lambda r: r["gap"])
        sticky_best[ss] = best
        print(f"{ss:5d} | {best['threshold']:8.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")

    # With FP tolerance
    print(f"\nWith FP ≤ 10%:")
    print(f"{'Start':>5} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 55)
    for ss in start_steps_to_try:
        results = analyze_sticky(rb_data, coco_data, ss, max(all_steps), thresholds)
        valid = [r for r in results if r["fp_rate"] <= 10.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            print(f"{ss:5d} | {best['threshold']:6.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")
        else:
            print(f"{ss:5d} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6}")

    # With FP tolerance ≤ 15%
    print(f"\nWith FP ≤ 15%:")
    print(f"{'Start':>5} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 55)
    for ss in start_steps_to_try:
        results = analyze_sticky(rb_data, coco_data, ss, max(all_steps), thresholds)
        valid = [r for r in results if r["fp_rate"] <= 15.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            print(f"{ss:5d} | {best['threshold']:6.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")
        else:
            print(f"{ss:5d} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6}")

    # ============================================
    # 4. Distribution at key steps
    # ============================================
    print("\n" + "=" * 80)
    print("4. P(harm) DISTRIBUTION AT KEY STEPS")
    print("=" * 80)

    key_steps = [0, 5, 10, 13, 15, 20, 25, 30, 35, 40, 45, 49]
    for step in key_steps:
        rb_scores = [s["steps"][step]["p_harm_max"] for s in rb_data if step in s["steps"]]
        coco_scores = [s["steps"][step]["p_harm_max"] for s in coco_data if step in s["steps"]]

        if rb_scores and coco_scores:
            print(f"\n  Step {step:2d}:")
            print(f"    RB  : mean={np.mean(rb_scores):.3f}, median={np.median(rb_scores):.3f}, "
                  f"min={np.min(rb_scores):.3f}, max={np.max(rb_scores):.3f}, "
                  f">0.1={sum(1 for x in rb_scores if x > 0.1)}/{len(rb_scores)}")
            print(f"    COCO: mean={np.mean(coco_scores):.3f}, median={np.median(coco_scores):.3f}, "
                  f"min={np.min(coco_scores):.3f}, max={np.max(coco_scores):.3f}, "
                  f">0.1={sum(1 for x in coco_scores if x > 0.1)}/{len(coco_scores)}")

    # ============================================
    # 5. Class-level breakdown
    # ============================================
    print("\n" + "=" * 80)
    print("5. CLASS-LEVEL BREAKDOWN (P(c2: nude) vs P(c3: color))")
    print("=" * 80)

    for step in [10, 13, 15, 20, 25, 30]:
        rb_c2 = [s["steps"][step]["p_c2"] for s in rb_data if step in s["steps"]]
        rb_c3 = [s["steps"][step]["p_c3"] for s in rb_data if step in s["steps"]]
        coco_c2 = [s["steps"][step]["p_c2"] for s in coco_data if step in s["steps"]]
        coco_c3 = [s["steps"][step]["p_c3"] for s in coco_data if step in s["steps"]]

        if rb_c2:
            print(f"\n  Step {step:2d}:")
            print(f"    RB   P(c2_nude):  mean={np.mean(rb_c2):.3f}, >0.1={sum(1 for x in rb_c2 if x>0.1)}/{len(rb_c2)}")
            print(f"    RB   P(c3_color): mean={np.mean(rb_c3):.3f}, >0.1={sum(1 for x in rb_c3 if x>0.1)}/{len(rb_c3)}")
            print(f"    COCO P(c2_nude):  mean={np.mean(coco_c2):.3f}, >0.1={sum(1 for x in coco_c2 if x>0.1)}/{len(coco_c2)}")
            print(f"    COCO P(c3_color): mean={np.mean(coco_c3):.3f}, >0.1={sum(1 for x in coco_c3 if x>0.1)}/{len(coco_c3)}")


if __name__ == "__main__":
    main()
