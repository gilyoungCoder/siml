#!/usr/bin/env python
"""
Analyze z0-trigger score gap between ringabell and COCO.
Compare with GradCAM-based monitoring results.
"""
import json
import numpy as np
from pathlib import Path


def load_z0_scores(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def analyze_single_step(rb_data, coco_data, step, thresholds, metric="p_harm"):
    results = []
    for thr in thresholds:
        tp_count = sum(1 for s in rb_data
                       if str(step) in s["z0_scores"] and s["z0_scores"][str(step)][metric] > thr)
        tp_rate = tp_count / len(rb_data) * 100

        fp_count = sum(1 for s in coco_data
                       if str(step) in s["z0_scores"] and s["z0_scores"][str(step)][metric] > thr)
        fp_rate = fp_count / len(coco_data) * 100

        gap = tp_rate - fp_rate
        results.append({"step": step, "threshold": thr,
                        "tp_rate": tp_rate, "fp_rate": fp_rate, "gap": gap,
                        "tp_count": tp_count, "fp_count": fp_count})
    return results


def analyze_sticky(rb_data, coco_data, start_step, end_step, thresholds, metric="p_harm"):
    results = []
    for thr in thresholds:
        tp_count = 0
        for s in rb_data:
            triggered = any(
                str(step) in s["z0_scores"] and s["z0_scores"][str(step)][metric] > thr
                for step in range(start_step, end_step + 1)
            )
            if triggered:
                tp_count += 1
        tp_rate = tp_count / len(rb_data) * 100

        fp_count = 0
        for s in coco_data:
            triggered = any(
                str(step) in s["z0_scores"] and s["z0_scores"][str(step)][metric] > thr
                for step in range(start_step, end_step + 1)
            )
            if triggered:
                fp_count += 1
        fp_rate = fp_count / len(coco_data) * 100

        gap = tp_rate - fp_rate
        results.append({"start_step": start_step, "end_step": end_step, "threshold": thr,
                        "tp_rate": tp_rate, "fp_rate": fp_rate, "gap": gap,
                        "tp_count": tp_count, "fp_count": fp_count})
    return results


def main():
    base = Path("/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/score_analysis_z0_trigger")
    rb_data = load_z0_scores(base / "ringabell_z0_scores.json")
    coco_data = load_z0_scores(base / "coco_z0_scores.json")

    print(f"Loaded: Ringabell={len(rb_data)}, COCO={len(coco_data)}")

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14,
                  0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ============================================
    # 1. Single-step analysis (best gap)
    # ============================================
    print("\n" + "=" * 80)
    print("1. SINGLE-STEP z0-TRIGGER ANALYSIS (P(harm) = P(nude) + P(color))")
    print("=" * 80)

    best_per_step = {}
    for step in range(50):
        results = analyze_single_step(rb_data, coco_data, step, thresholds)
        best = max(results, key=lambda r: r["gap"])
        best_per_step[step] = best

    print(f"\n{'Step':>4} | {'Best Thr':>8} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 50)
    for step in range(50):
        b = best_per_step[step]
        print(f"{step:4d} | {b['threshold']:8.2f} | {b['tp_rate']:6.1f}% | {b['fp_rate']:7.1f}% | {b['gap']:+6.1f}%")

    sorted_steps = sorted(best_per_step.items(), key=lambda x: x[1]["gap"], reverse=True)
    print(f"\nTop 10 steps by gap:")
    for step, b in sorted_steps[:10]:
        print(f"  Step {step:2d}: thr={b['threshold']:.2f} → RB {b['tp_rate']:.1f}%, COCO {b['fp_rate']:.1f}%, gap={b['gap']:+.1f}%")

    # ============================================
    # 2. With FP tolerance
    # ============================================
    print("\n" + "=" * 80)
    print("2. BEST THRESHOLDS WITH FP TOLERANCE (COCO FP ≤ 10%)")
    print("=" * 80)

    print(f"\n{'Step':>4} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 50)
    for step in range(50):
        results = analyze_single_step(rb_data, coco_data, step, thresholds)
        valid = [r for r in results if r["fp_rate"] <= 10.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            print(f"{step:4d} | {best['threshold']:6.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")
        else:
            print(f"{step:4d} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6}")

    # ============================================
    # 3. Sticky trigger
    # ============================================
    print("\n" + "=" * 80)
    print("3. STICKY TRIGGER ANALYSIS (start_step..49)")
    print("=" * 80)

    start_steps = [0, 3, 5, 7, 10, 13, 15, 20, 25, 30]

    print(f"\n{'Start':>5} | {'Best Thr':>8} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 55)
    for ss in start_steps:
        results = analyze_sticky(rb_data, coco_data, ss, 49, thresholds)
        best = max(results, key=lambda r: r["gap"])
        print(f"{ss:5d} | {best['threshold']:8.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")

    print(f"\nWith FP ≤ 10%:")
    print(f"{'Start':>5} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 55)
    for ss in start_steps:
        results = analyze_sticky(rb_data, coco_data, ss, 49, thresholds)
        valid = [r for r in results if r["fp_rate"] <= 10.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            print(f"{ss:5d} | {best['threshold']:6.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")
        else:
            print(f"{ss:5d} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6}")

    print(f"\nWith FP ≤ 15%:")
    print(f"{'Start':>5} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 55)
    for ss in start_steps:
        results = analyze_sticky(rb_data, coco_data, ss, 49, thresholds)
        valid = [r for r in results if r["fp_rate"] <= 15.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            print(f"{ss:5d} | {best['threshold']:6.2f} | {best['tp_rate']:6.1f}% | {best['fp_rate']:7.1f}% | {best['gap']:+6.1f}%")
        else:
            print(f"{ss:5d} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>6}")

    # ============================================
    # 4. Distribution
    # ============================================
    print("\n" + "=" * 80)
    print("4. P(harm) DISTRIBUTION AT KEY STEPS")
    print("=" * 80)

    for step in [0, 5, 10, 13, 15, 20, 25, 30, 35, 40, 45, 49]:
        rb_scores = [s["z0_scores"][str(step)]["p_harm"] for s in rb_data if str(step) in s["z0_scores"]]
        coco_scores = [s["z0_scores"][str(step)]["p_harm"] for s in coco_data if str(step) in s["z0_scores"]]
        if rb_scores and coco_scores:
            print(f"\n  Step {step:2d}:")
            print(f"    RB  : mean={np.mean(rb_scores):.3f}, median={np.median(rb_scores):.3f}, "
                  f"min={np.min(rb_scores):.3f}, max={np.max(rb_scores):.3f}, "
                  f">0.1={sum(1 for x in rb_scores if x > 0.1)}/{len(rb_scores)}")
            print(f"    COCO: mean={np.mean(coco_scores):.3f}, median={np.median(coco_scores):.3f}, "
                  f"min={np.min(coco_scores):.3f}, max={np.max(coco_scores):.3f}, "
                  f">0.1={sum(1 for x in coco_scores if x > 0.1)}/{len(coco_scores)}")

    # ============================================
    # 5. Comparison with GradCAM
    # ============================================
    print("\n" + "=" * 80)
    print("5. COMPARISON: z0-TRIGGER vs GradCAM (best single-step, FP≤10%)")
    print("=" * 80)

    print(f"\n{'Method':<20} | {'Step':>4} | {'Thr':>6} | {'RB TP%':>7} | {'COCO FP%':>8} | {'Gap':>6}")
    print("-" * 70)

    # z0-trigger best (FP≤10%)
    best_z0 = None
    for step in range(50):
        results = analyze_single_step(rb_data, coco_data, step, thresholds)
        valid = [r for r in results if r["fp_rate"] <= 10.0]
        if valid:
            best = max(valid, key=lambda r: r["tp_rate"])
            if best_z0 is None or best["tp_rate"] > best_z0["tp_rate"]:
                best_z0 = {**best, "step": step}

    if best_z0:
        print(f"{'z0-trigger':<20} | {best_z0['step']:4d} | {best_z0['threshold']:6.2f} | "
              f"{best_z0['tp_rate']:6.1f}% | {best_z0['fp_rate']:7.1f}% | {best_z0['gap']:+6.1f}%")

    # Reference values from previous GradCAM analysis
    print(f"{'GradCAM (previous)':<20} | {'24':>4} | {'0.35':>6} | {'81.0':>6}% | {'8.0':>7}% | {'+73.0':>6}%")
    print(f"{'z0 ResNet18 (ref)':<20} | {'13':>4} | {'0.08':>6} | {'92.4':>6}% | {'8.0':>7}% | {'+84.4':>6}%")


if __name__ == "__main__":
    main()
