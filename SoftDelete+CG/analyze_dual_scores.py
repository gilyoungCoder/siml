#!/usr/bin/env python
"""
Analyze dual scores (z0 + GradCAM CDF) from score_analysis_dual.py output.

Simulates z0_single monitoring: at decision_step, check z0 p_harm.
If triggered, check how many subsequent steps GradCAM CDF would guide.

Reports:
  1. Per-(decision_step, z0_thr): z0 trigger rate (RB vs COCO)
  2. Per-(decision_step, z0_thr, cdf_thr): effective guidance rate after trigger
  3. Best configs with high RB detection and low COCO FP

Usage:
    python analyze_dual_scores.py \
        --ringabell scg_outputs/score_analysis_dual/ringabell_scores.json \
        --coco scg_outputs/score_analysis_dual/coco_scores.json

    # With split files:
    python analyze_dual_scores.py \
        --ringabell scg_outputs/score_analysis_dual/ringabell_scores_A.json \
                    scg_outputs/score_analysis_dual/ringabell_scores_B.json \
                    scg_outputs/score_analysis_dual/ringabell_scores_C.json \
        --coco scg_outputs/score_analysis_dual/coco_scores.json
"""
import json
from argparse import ArgumentParser


def load_scores(paths):
    """Load and merge results from one or more score JSON files."""
    if isinstance(paths, str):
        paths = [paths]
    all_results = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data["results"])
    return all_results


def analyze_z0_trigger(results, decision_step, z0_thr):
    """At exactly decision_step, how many prompts have z0_p_harm > z0_thr?"""
    triggered = 0
    total = len(results)
    for r in results:
        step_key = str(decision_step)
        if step_key in r["scores"]:
            if r["scores"][step_key]["z0_p_harm"] > z0_thr:
                triggered += 1
    return triggered, total


def analyze_cdf_after_trigger(results, decision_step, z0_thr, cdf_thr,
                              num_steps=50):
    """For prompts that z0-trigger at decision_step, simulate GradCAM CDF guidance.

    Returns:
        triggered: number of prompts where z0 triggers
        avg_guided_pct: average % of post-trigger steps where CDF guides
    """
    triggered = 0
    guided_pcts = []

    for r in results:
        step_key = str(decision_step)
        if step_key not in r["scores"]:
            continue

        z0_harm = r["scores"][step_key]["z0_p_harm"]
        if z0_harm <= z0_thr:
            continue

        triggered += 1

        # Count GradCAM CDF guided steps after trigger
        guided = 0
        total_post = 0
        for s in range(decision_step, num_steps):
            sk = str(s)
            if sk not in r["scores"]:
                continue
            total_post += 1
            sc = r["scores"][sk]
            # CDF guide if either class exceeds threshold
            cdf_max = max(sc.get("cdf_p_nude", 0), sc.get("cdf_p_color", 0))
            if cdf_max > cdf_thr:
                guided += 1

        if total_post > 0:
            guided_pcts.append(guided / total_post * 100)

    avg_guided = sum(guided_pcts) / len(guided_pcts) if guided_pcts else 0
    return triggered, avg_guided


def main():
    parser = ArgumentParser()
    parser.add_argument("--ringabell", type=str, nargs="+", required=True)
    parser.add_argument("--coco", type=str, nargs="+", required=True)
    args = parser.parse_args()

    rb = load_scores(args.ringabell)
    coco = load_scores(args.coco)
    print(f"Ringabell: {len(rb)} prompts, COCO: {len(coco)} prompts\n")

    # ======================================================
    # 1. z0 trigger rate per (decision_step, z0_thr)
    # ======================================================
    decision_steps = [3, 5, 7, 8, 10, 12, 15]
    z0_thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]

    print("=" * 100)
    print("z0 TRIGGER RATE: Ringabell vs COCO")
    print("=" * 100)
    print(f"{'Step':>4} |", end="")
    for thr in z0_thresholds:
        print(f"  T>{thr:.2f} (RB/CO)", end="")
    print()
    print("-" * 100)

    for ds in decision_steps:
        print(f"{ds:>4} |", end="")
        for thr in z0_thresholds:
            rb_trig, rb_tot = analyze_z0_trigger(rb, ds, thr)
            co_trig, co_tot = analyze_z0_trigger(coco, ds, thr)
            rb_pct = rb_trig / rb_tot * 100 if rb_tot else 0
            co_pct = co_trig / co_tot * 100 if co_tot else 0
            print(f"  {rb_pct:>4.0f}/{co_pct:<4.0f}     ", end="")
        print()

    # ======================================================
    # 2. Best z0 configs (COCO FP = 0%)
    # ======================================================
    print()
    print("=" * 80)
    print("BEST Z0 CONFIGS (COCO FP = 0%)")
    print("=" * 80)
    print(f"{'Step':>4} | {'z0_thr':>6} | {'RB trigger':>10} | {'COCO FP':>8}")
    print("-" * 45)

    best_z0 = []
    for ds in range(50):
        for thr in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]:
            rb_trig, rb_tot = analyze_z0_trigger(rb, ds, thr)
            co_trig, co_tot = analyze_z0_trigger(coco, ds, thr)
            rb_pct = rb_trig / rb_tot * 100 if rb_tot else 0
            co_pct = co_trig / co_tot * 100 if co_tot else 0
            if co_trig == 0 and rb_pct > 50:
                best_z0.append((ds, thr, rb_pct, co_pct, rb_trig, rb_tot))

    best_z0.sort(key=lambda x: -x[2])
    for ds, thr, rb_pct, co_pct, rb_trig, rb_tot in best_z0[:20]:
        print(f"{ds:>4} | {thr:>6.2f} | {rb_trig:>3}/{rb_tot} ({rb_pct:>5.1f}%) | {co_pct:>6.1f}%")

    # ======================================================
    # 3. Full dual simulation: z0 trigger + GradCAM CDF
    # ======================================================
    cdf_thresholds = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]

    print()
    print("=" * 120)
    print("DUAL SIMULATION: z0 trigger → GradCAM CDF guidance")
    print("(For each triggered prompt, avg % of post-trigger steps guided by CDF)")
    print("=" * 120)

    # Pick top z0 configs to analyze further
    top_z0_configs = [(7, 0.02), (7, 0.03), (8, 0.02), (8, 0.03),
                      (5, 0.03), (5, 0.05), (10, 0.03), (10, 0.05)]

    print(f"{'Config':>20} | {'RB z0-trig':>10} | {'CO z0-trig':>10} |",
          end="")
    for ct in cdf_thresholds:
        print(f" CDF>{ct:.2f}(RB%)", end="")
    print()
    print("-" * 120)

    for ds, z0t in top_z0_configs:
        rb_trig, rb_tot = analyze_z0_trigger(rb, ds, z0t)
        co_trig, co_tot = analyze_z0_trigger(coco, ds, z0t)
        rb_pct = rb_trig / rb_tot * 100 if rb_tot else 0
        co_pct = co_trig / co_tot * 100 if co_tot else 0

        config_str = f"step={ds} z0>{z0t:.2f}"
        print(f"{config_str:>20} | "
              f"{rb_trig:>3}/{rb_tot} ({rb_pct:>4.0f}%) | "
              f"{co_trig:>3}/{co_tot} ({co_pct:>4.0f}%) |", end="")

        for ct in cdf_thresholds:
            _, rb_guided = analyze_cdf_after_trigger(rb, ds, z0t, ct)
            print(f"   {rb_guided:>5.1f}%   ", end="")
        print()

    # ======================================================
    # 4. Per-step GradCAM CDF distribution (for reference)
    # ======================================================
    print()
    print("=" * 100)
    print("GradCAM CDF: per-step distribution (all prompts)")
    print("=" * 100)
    print(f"{'Step':>4} | {'RB cdf_nude mean':>16} {'RB cdf_nude max':>15} "
          f"| {'CO cdf_nude mean':>16} {'CO cdf_nude max':>15}")
    print("-" * 80)

    for step in [0, 3, 5, 7, 8, 10, 15, 20, 30, 49]:
        sk = str(step)
        rb_vals = [r["scores"][sk]["cdf_p_nude"]
                   for r in rb if sk in r["scores"]]
        co_vals = [r["scores"][sk]["cdf_p_nude"]
                   for r in coco if sk in r["scores"]]

        if rb_vals and co_vals:
            import numpy as np
            rb_arr = np.array(rb_vals)
            co_arr = np.array(co_vals)
            print(f"{step:>4} | {rb_arr.mean():>16.4f} {rb_arr.max():>15.4f} "
                  f"| {co_arr.mean():>16.4f} {co_arr.max():>15.4f}")

    print()


if __name__ == "__main__":
    main()
