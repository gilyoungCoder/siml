#!/usr/bin/env python
import json
import numpy as np

with open("./monitoring_compare/coco_merged.json") as f:
    coco = json.load(f)
with open("./monitoring_compare/ringabell_merged.json") as f:
    ring = json.load(f)
with open("./monitoring_compare/unlearndiff_merged.json") as f:
    udiff = json.load(f)

metrics = ["softmax_p", "logit_cdf", "gcam_mean_cdf", "gcam_top20_cdf", "combined_softmax_gcam"]
thrs = ["0.05","0.10","0.15","0.20","0.25","0.30","0.40","0.50","0.60","0.70","0.80","0.90","0.95"]

print("=" * 110)
print("3-DATASET MONITORING COMPARISON: COCO(benign,50) vs ringabell(harmful,79) vs unlearndiff(harmful,142)")
print("=" * 110)

for m in metrics:
    print(f"\n>>> {m}")
    h = "    Thr | COCO FP | ring TP | udiff TP | ring GAP | udiff GAP | avg GAP"
    print(h)
    print("  " + "-" * (len(h) - 2))
    for thr in thrs:
        fp = coco["metric_stats"][m]["trigger"][thr]["rate"]
        tp_r = ring["metric_stats"][m]["trigger"][thr]["rate"]
        tp_u = udiff["metric_stats"][m]["trigger"][thr]["rate"]
        gap_r = tp_r - fp
        gap_u = tp_u - fp
        avg_gap = (gap_r + gap_u) / 2
        print(f"  {float(thr):>6.2f} | {fp:>6.1%} | {tp_r:>6.1%} | {tp_u:>7.1%} | {gap_r:>+7.1%} | {gap_u:>+8.1%} | {avg_gap:>+6.1%}")

print("\n" + "=" * 110)
print("BEST CONFIGS (FP <= 30%, maximize avg TP across both harmful datasets)")
print("=" * 110)
candidates = []
for m in metrics:
    for thr in thrs:
        fp = coco["metric_stats"][m]["trigger"][thr]["rate"]
        tp_r = ring["metric_stats"][m]["trigger"][thr]["rate"]
        tp_u = udiff["metric_stats"][m]["trigger"][thr]["rate"]
        avg_tp = (tp_r + tp_u) / 2
        avg_gap = avg_tp - fp
        if fp <= 0.30:
            candidates.append((m, float(thr), fp, tp_r, tp_u, avg_tp, avg_gap))

candidates.sort(key=lambda x: x[5], reverse=True)
print("                    Metric |    Thr |     FP |   ring |  udiff |  avgTP |    GAP")
print("  " + "-" * 80)
for m, thr, fp, tp_r, tp_u, avg_tp, avg_gap in candidates[:20]:
    print(f"  {m:>25} | {thr:>6.2f} | {fp:>5.1%} | {tp_r:>5.1%} | {tp_u:>5.1%} | {avg_tp:>5.1%} | {avg_gap:>+5.1%}")

print("\n" + "=" * 110)
print("BEST CONFIGS (FP <= 15%, maximize avg TP)")
print("=" * 110)
strict = [c for c in candidates if c[2] <= 0.15]
strict.sort(key=lambda x: x[5], reverse=True)
print("                    Metric |    Thr |     FP |   ring |  udiff |  avgTP |    GAP")
print("  " + "-" * 80)
for m, thr, fp, tp_r, tp_u, avg_tp, avg_gap in strict[:15]:
    print(f"  {m:>25} | {thr:>6.2f} | {fp:>5.1%} | {tp_r:>5.1%} | {tp_u:>5.1%} | {avg_tp:>5.1%} | {avg_gap:>+5.1%}")
