#!/usr/bin/env python3
"""WHEN(CAS) trigger-rate analysis for SD3 + FLUX1.0.

Reads `generation_stats.json` from existing ours runs and reports
how often CAS exceeds threshold τ on RAB (should: high TP) and
COCO (should: low FP).
"""
import json
from pathlib import Path

ROOT = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs")

SD3_RUNS = {
    "RAB":  ROOT / "sd3/safegen/rab/generation_stats.json",
    "COCO": ROOT / "sd3/safegen/coco250/generation_stats.json",
}

FLUX_RUNS = {
    "RAB":  ROOT / "flux1dev_experiments/ours/rab_single_ainp_ss1.0_cas0.6/generation_stats.json",
    "COCO": ROOT / "flux1dev_experiments/ours/coco_family_ainp_ss1.0_cas0.6/generation_stats.json",
}

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load(p):
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def stats(values, thresholds):
    n = len(values)
    if n == 0:
        return None
    sv = sorted(values)
    mean = sum(values) / n
    median = sv[n // 2]
    p25, p75 = sv[n // 4], sv[3 * n // 4]
    rates = {t: sum(v > t for v in values) / n for t in thresholds}
    return n, mean, median, p25, p75, rates


def report(name, runs):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    print(f"{'Dataset':10s} {'N':>4s} {'mean':>6s} {'median':>7s} "
          f"{'P25':>6s} {'P75':>6s}  " +
          "  ".join(f"τ>{t:.1f}" for t in THRESHOLDS))
    print('-' * 72)
    summary = {}
    for ds, fp in runs.items():
        data = load(fp)
        if not data:
            print(f"{ds:10s}  MISSING: {fp}")
            continue
        vals = [d["max_cas"] for d in data]
        n, m, med, p25, p75, rates = stats(vals, THRESHOLDS)
        rate_str = "  ".join(f"{rates[t]*100:>4.1f}%" for t in THRESHOLDS)
        print(f"{ds:10s} {n:>4d} {m:>6.3f} {med:>7.3f} {p25:>6.3f} {p75:>6.3f}  {rate_str}")
        summary[ds] = {"n": n, "mean": m, "median": med, "rates": rates,
                       "values": vals}
    return summary


def gap_analysis(name, summary):
    if "RAB" not in summary or "COCO" not in summary:
        return
    rab = summary["RAB"]["rates"]
    coco = summary["COCO"]["rates"]
    print(f"\n  {name} TP-FP gap (TP_RAB - FP_COCO):")
    print(f"  {'τ':>4s}  {'TP(RAB)':>8s}  {'FP(COCO)':>9s}  {'Gap':>7s}")
    for t in THRESHOLDS:
        gap = (rab[t] - coco[t]) * 100
        print(f"  {t:>4.1f}  {rab[t]*100:>7.1f}%  {coco[t]*100:>8.1f}%  {gap:>+6.1f}pp")


if __name__ == "__main__":
    sd3 = report("SD 3.0 — text probe, ainp, cas_thr=0.4 (eval τ varies)", SD3_RUNS)
    gap_analysis("SD 3.0", sd3)
    flux = report("FLUX 1.0-dev — single ainp ss=1.0 cas_thr=0.6 (eval τ varies)", FLUX_RUNS)
    gap_analysis("FLUX 1.0-dev", flux)
    print()
