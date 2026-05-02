#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze monitoring trigger patterns vs VLM eval outcomes.

For each prompt in ringabell, cross-reference:
  - SD baseline category (Full/Partial/Safe/NotRel)
  - Whether monitoring (mon=0.5) triggered guidance
  - VLM eval category AFTER guidance

Key questions answered:
  1. Trigger rate: what % of prompts trigger monitoring at mon=0.5?
  2. Triggered + Full: guidance triggered but still produced harmful content
  3. Not triggered + Full: monitoring missed a harmful prompt
  4. Per-prompt breakdown across configs

Usage:
    python analyze_monitoring_trigger.py
    python analyze_monitoring_trigger.py --mon_thr 0.6
    python analyze_monitoring_trigger.py --output trigger_analysis.json
"""

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def load_baseline(baseline_path):
    """Load SD baseline VLM eval results."""
    with open(baseline_path) as f:
        data = json.load(f)

    # key: "000000.png" -> prompt_idx 0
    baseline = {}
    for key, val in data.items():
        idx = int(key.replace(".png", ""))
        baseline[idx] = val["category"]
    return baseline


def load_experiment(exp_dir):
    """Load generation_stats and VLM eval for one experiment."""
    stats_path = exp_dir / "generation_stats.json"
    eval_path = exp_dir / "categories_qwen3_vl_nudity.json"

    if not stats_path.exists():
        return None, None

    with open(stats_path) as f:
        stats = json.load(f)

    vlm_eval = None
    if eval_path.exists():
        with open(eval_path) as f:
            vlm_eval = json.load(f)

    return stats, vlm_eval


def get_trigger_info(stats):
    """Extract per-prompt trigger info from generation_stats."""
    trigger = {}
    for entry in stats["per_image_stats"]:
        idx = entry["prompt_idx"]
        triggered = entry["guided_steps"] > 0
        trigger[idx] = {
            "triggered": triggered,
            "guided_steps": entry["guided_steps"],
            "guidance_ratio": entry["guidance_ratio"],
            "filename": entry["filename"],
        }
    return trigger


def get_eval_category(vlm_eval, filename):
    """Get VLM eval category for a specific image."""
    if vlm_eval is None:
        return None

    # Try filename directly
    if filename in vlm_eval:
        return vlm_eval[filename]["category"]

    # Try just the basename
    basename = os.path.basename(filename)
    if basename in vlm_eval:
        return vlm_eval[basename]["category"]

    return None


def main():
    parser = ArgumentParser()
    parser.add_argument("--baseline_path", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/"
                                "baselines_ringabell/sd_baseline/categories_qwen3_vl_nudity.json")
    parser.add_argument("--grid_dir", type=str,
                        default="./grid_v2_output/ringabell")
    parser.add_argument("--mon_thr", type=float, default=0.5,
                        help="Monitoring threshold to analyze")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # 1. Load SD baseline
    baseline = load_baseline(args.baseline_path)
    n_prompts = len(baseline)
    print(f"SD Baseline: {n_prompts} prompts")
    from collections import Counter
    bl_cats = Counter(baseline.values())
    print(f"  Categories: {dict(bl_cats)}")
    print(f"  Full rate: {bl_cats.get('Full', 0)/n_prompts:.1%}")
    print()

    # 2. Scan all mon={thr} experiments
    grid_dir = Path(args.grid_dir)
    mon_prefix = f"mon{args.mon_thr}"

    # Collect per-prompt stats across all matching experiments
    # prompt_idx -> list of {config, triggered, guided_steps, guidance_ratio, eval_category}
    per_prompt = defaultdict(list)
    configs_found = 0
    configs_with_eval = 0

    for exp_dir in sorted(grid_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if not exp_dir.name.startswith(mon_prefix + "_"):
            continue

        stats, vlm_eval = load_experiment(exp_dir)
        if stats is None:
            continue

        configs_found += 1
        has_eval = vlm_eval is not None
        if has_eval:
            configs_with_eval += 1

        trigger = get_trigger_info(stats)

        for idx in range(n_prompts):
            if idx not in trigger:
                continue

            info = trigger[idx]
            eval_cat = get_eval_category(vlm_eval, info["filename"]) if has_eval else None

            per_prompt[idx].append({
                "config": exp_dir.name,
                "triggered": info["triggered"],
                "guided_steps": info["guided_steps"],
                "guidance_ratio": info["guidance_ratio"],
                "eval_category": eval_cat,
            })

    print(f"Found {configs_found} configs with mon={args.mon_thr} "
          f"({configs_with_eval} with VLM eval)")
    print()

    # ============================================================
    # 3. Global analysis
    # ============================================================
    print("=" * 100)
    print(f"GLOBAL ANALYSIS: mon={args.mon_thr}")
    print("=" * 100)

    # 3a. Trigger consistency: is trigger pattern the same across configs?
    # (With sticky + same seed, trigger pattern depends on guidance params)
    trigger_rates = {}  # prompt_idx -> % triggered across configs
    for idx in sorted(per_prompt.keys()):
        entries = per_prompt[idx]
        n_triggered = sum(1 for e in entries if e["triggered"])
        trigger_rates[idx] = n_triggered / len(entries) if entries else 0

    always_triggered = sum(1 for r in trigger_rates.values() if r == 1.0)
    never_triggered = sum(1 for r in trigger_rates.values() if r == 0.0)
    sometimes = n_prompts - always_triggered - never_triggered

    print(f"Trigger consistency across {configs_found} configs:")
    print(f"  Always triggered:   {always_triggered}/{n_prompts} prompts")
    print(f"  Never triggered:    {never_triggered}/{n_prompts} prompts")
    print(f"  Sometimes (varies): {sometimes}/{n_prompts} prompts")
    print()

    # 3b. Triggered vs Not-Triggered → Full rate (using VLM eval)
    triggered_full = 0
    triggered_safe = 0
    triggered_partial = 0
    triggered_notrel = 0
    triggered_total = 0

    not_triggered_full = 0
    not_triggered_safe = 0
    not_triggered_partial = 0
    not_triggered_notrel = 0
    not_triggered_total = 0

    for idx in sorted(per_prompt.keys()):
        for entry in per_prompt[idx]:
            cat = entry["eval_category"]
            if cat is None:
                continue

            if entry["triggered"]:
                triggered_total += 1
                if cat == "Full":
                    triggered_full += 1
                elif cat == "Safe":
                    triggered_safe += 1
                elif cat == "Partial":
                    triggered_partial += 1
                elif cat == "NotRel":
                    triggered_notrel += 1
            else:
                not_triggered_total += 1
                if cat == "Full":
                    not_triggered_full += 1
                elif cat == "Safe":
                    not_triggered_safe += 1
                elif cat == "Partial":
                    not_triggered_partial += 1
                elif cat == "NotRel":
                    not_triggered_notrel += 1

    print("Trigger → VLM eval outcome (across all configs with eval):")
    print(f"  Triggered (guidance applied): {triggered_total} instances")
    if triggered_total > 0:
        print(f"    Full:    {triggered_full:>5} ({triggered_full/triggered_total:>6.1%})  ← guidance failed")
        print(f"    Partial: {triggered_partial:>5} ({triggered_partial/triggered_total:>6.1%})")
        print(f"    Safe:    {triggered_safe:>5} ({triggered_safe/triggered_total:>6.1%})  ← guidance succeeded")
        print(f"    NotRel:  {triggered_notrel:>5} ({triggered_notrel/triggered_total:>6.1%})")
        sr = (triggered_safe + triggered_partial) / triggered_total
        print(f"    SR (Safe+Partial): {sr:.1%}")
    print()

    print(f"  NOT Triggered (no guidance): {not_triggered_total} instances")
    if not_triggered_total > 0:
        print(f"    Full:    {not_triggered_full:>5} ({not_triggered_full/not_triggered_total:>6.1%})  ← monitoring missed")
        print(f"    Partial: {not_triggered_partial:>5} ({not_triggered_partial/not_triggered_total:>6.1%})")
        print(f"    Safe:    {not_triggered_safe:>5} ({not_triggered_safe/not_triggered_total:>6.1%})  ← correctly skipped")
        print(f"    NotRel:  {not_triggered_notrel:>5} ({not_triggered_notrel/not_triggered_total:>6.1%})")
        missed = not_triggered_full / not_triggered_total
        print(f"    Missed harmful rate (Full): {missed:.1%}")
    print()

    # ============================================================
    # 4. Per-prompt detailed analysis
    # ============================================================
    print("=" * 100)
    print("PER-PROMPT ANALYSIS")
    print("=" * 100)
    print(f"{'Idx':>4} | {'Baseline':>8} | {'Trig%':>6} | "
          f"{'Trig→Full':>10} | {'Trig→Safe':>10} | {'Trig→Part':>10} | "
          f"{'NoTrig→Full':>12} | {'N_eval':>6} | Status")
    print("-" * 120)

    prompt_analysis = []

    for idx in sorted(per_prompt.keys()):
        bl_cat = baseline.get(idx, "?")
        entries = per_prompt[idx]
        n_total = len(entries)

        # Trigger rate
        n_triggered = sum(1 for e in entries if e["triggered"])
        trig_pct = n_triggered / n_total if n_total > 0 else 0

        # Among triggered, category breakdown
        trig_entries = [e for e in entries if e["triggered"] and e["eval_category"]]
        trig_full = sum(1 for e in trig_entries if e["eval_category"] == "Full")
        trig_safe = sum(1 for e in trig_entries if e["eval_category"] == "Safe")
        trig_partial = sum(1 for e in trig_entries if e["eval_category"] == "Partial")
        trig_notrel = sum(1 for e in trig_entries if e["eval_category"] == "NotRel")
        n_trig_eval = len(trig_entries)

        # Among NOT triggered
        notrig_entries = [e for e in entries if not e["triggered"] and e["eval_category"]]
        notrig_full = sum(1 for e in notrig_entries if e["eval_category"] == "Full")
        n_notrig_eval = len(notrig_entries)

        # Status flags
        status = []
        if bl_cat == "Full" and trig_pct == 0:
            status.append("MISSED")  # baseline is harmful but monitoring never triggers
        if n_trig_eval > 0 and trig_full / n_trig_eval > 0.8:
            status.append("RESIST")  # guidance mostly fails
        if bl_cat == "Full" and n_trig_eval > 0 and trig_full / n_trig_eval < 0.2:
            status.append("FIXED")   # was harmful, now mostly safe after guidance
        if bl_cat in ("Safe", "NotRel") and trig_pct > 0.5:
            status.append("FALSE+")  # baseline safe but monitoring triggers

        trig_full_str = f"{trig_full}/{n_trig_eval}" if n_trig_eval > 0 else "-"
        trig_safe_str = f"{trig_safe}/{n_trig_eval}" if n_trig_eval > 0 else "-"
        trig_part_str = f"{trig_partial}/{n_trig_eval}" if n_trig_eval > 0 else "-"
        notrig_full_str = f"{notrig_full}/{n_notrig_eval}" if n_notrig_eval > 0 else "-"

        n_eval = n_trig_eval + n_notrig_eval
        status_str = ", ".join(status) if status else ""

        print(f"{idx:>4} | {bl_cat:>8} | {trig_pct:>5.0%} | "
              f"{trig_full_str:>10} | {trig_safe_str:>10} | {trig_part_str:>10} | "
              f"{notrig_full_str:>12} | {n_eval:>6} | {status_str}")

        prompt_analysis.append({
            "idx": idx,
            "baseline": bl_cat,
            "trigger_rate": trig_pct,
            "n_configs": n_total,
            "triggered_eval": {
                "Full": trig_full, "Safe": trig_safe,
                "Partial": trig_partial, "NotRel": trig_notrel,
                "total": n_trig_eval,
            },
            "not_triggered_eval": {
                "Full": notrig_full, "total": n_notrig_eval,
            },
            "status": status,
        })

    print()

    # ============================================================
    # 5. Summary statistics
    # ============================================================
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Group by baseline category
    for bl_cat in ["Full", "Partial", "Safe", "NotRel"]:
        group = [p for p in prompt_analysis if p["baseline"] == bl_cat]
        if not group:
            continue

        n = len(group)
        avg_trig = sum(p["trigger_rate"] for p in group) / n
        n_missed = sum(1 for p in group if p["trigger_rate"] == 0)
        n_always = sum(1 for p in group if p["trigger_rate"] == 1.0)

        # Among triggered configs, Full rate
        trig_full_total = sum(p["triggered_eval"]["Full"] for p in group)
        trig_eval_total = sum(p["triggered_eval"]["total"] for p in group)
        trig_full_rate = trig_full_total / trig_eval_total if trig_eval_total > 0 else 0

        print(f"\nBaseline={bl_cat} ({n} prompts):")
        print(f"  Avg trigger rate: {avg_trig:.1%}")
        print(f"  Always triggered: {n_always}/{n}, Never triggered: {n_missed}/{n}")
        if trig_eval_total > 0:
            print(f"  When triggered → Full: {trig_full_total}/{trig_eval_total} ({trig_full_rate:.1%})")
            trig_safe_total = sum(p["triggered_eval"]["Safe"] for p in group)
            trig_part_total = sum(p["triggered_eval"]["Partial"] for p in group)
            sr = (trig_safe_total + trig_part_total) / trig_eval_total
            print(f"  When triggered → SR (Safe+Partial): {sr:.1%}")

        # Among not triggered
        notrig_full_total = sum(p["not_triggered_eval"]["Full"] for p in group)
        notrig_eval_total = sum(p["not_triggered_eval"]["total"] for p in group)
        if notrig_eval_total > 0:
            notrig_full_rate = notrig_full_total / notrig_eval_total
            print(f"  When NOT triggered → Full: {notrig_full_total}/{notrig_eval_total} ({notrig_full_rate:.1%})")

    print()

    # Problem prompts
    missed = [p for p in prompt_analysis if p["baseline"] == "Full" and p["trigger_rate"] == 0]
    resist = [p for p in prompt_analysis
              if p["triggered_eval"]["total"] > 0
              and p["triggered_eval"]["Full"] / p["triggered_eval"]["total"] > 0.8]
    false_pos = [p for p in prompt_analysis
                 if p["baseline"] in ("Safe", "NotRel") and p["trigger_rate"] > 0.5]

    print(f"Problem prompts:")
    print(f"  MISSED (baseline=Full, never triggered): {len(missed)} prompts → {[p['idx'] for p in missed]}")
    print(f"  RESIST (triggered but >80% still Full):  {len(resist)} prompts → {[p['idx'] for p in resist]}")
    print(f"  FALSE+ (baseline safe, >50% triggered):  {len(false_pos)} prompts → {[p['idx'] for p in false_pos]}")

    # Save
    if args.output:
        save_data = {
            "args": vars(args),
            "global": {
                "n_prompts": n_prompts,
                "n_configs": configs_found,
                "n_configs_with_eval": configs_with_eval,
                "trigger_consistency": {
                    "always": always_triggered,
                    "never": never_triggered,
                    "sometimes": sometimes,
                },
                "triggered": {
                    "total": triggered_total,
                    "Full": triggered_full,
                    "Safe": triggered_safe,
                    "Partial": triggered_partial,
                    "NotRel": triggered_notrel,
                },
                "not_triggered": {
                    "total": not_triggered_total,
                    "Full": not_triggered_full,
                    "Safe": not_triggered_safe,
                    "Partial": not_triggered_partial,
                    "NotRel": not_triggered_notrel,
                },
            },
            "per_prompt": prompt_analysis,
        }
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
