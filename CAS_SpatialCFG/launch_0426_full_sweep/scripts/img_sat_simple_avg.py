#!/usr/bin/env python3
"""Simple single-line K-saturation plot averaged across concepts.
Uses existing nested data — NO fabrication.
"""
import csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures")
SRC = OUT / "img_sat_nested_table.csv"

rows = list(csv.DictReader(open(SRC)))
KS = [1, 2, 4, 8, 12, 16]

# Aggregate across concepts: SR / Full / NR per K
def agg(metric):
    by_K = collections.defaultdict(list)
    for r in rows:
        if r[metric] in ("", "None"): continue
        by_K[int(r["K"])].append(float(r[metric]))
    return {k: sum(v)/len(v) for k,v in by_K.items()}

sr_avg   = agg("SR_pct")
full_avg = agg("Full_pct")
nr_avg   = agg("NR_pct")

print("K  | avg SR | avg Full% | avg NR%")
for k in KS:
    print(f"{k:>2} | {sr_avg[k]:>6.1f} | {full_avg[k]:>8.1f}  | {nr_avg[k]:>6.1f}")

# Plot: 3 panels (SR / Full / NR) avg-across-concept
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, metric, data, color, label, ylim in [
    (axes[0], "SR",   sr_avg,   "#d62728", "SR (Safe+Partial) %",     (50, 70)),
    (axes[1], "Full", full_avg, "#ff7f0e", "Full violation %",          (15, 30)),
    (axes[2], "NR",   nr_avg,   "#1f77b4", "NotRelevant %",             (5, 25)),
]:
    xs = KS
    ys = [data[k] for k in KS]
    ax.plot(xs, ys, "-o", color=color, linewidth=2.5, markersize=8)
    ax.axvline(x=4, linestyle="--", color="gray", alpha=0.5, label="K=4 (paper default)")
    # highlight K=4 marker
    ax.plot([4], [data[4]], "o", markersize=14, markerfacecolor="none", markeredgecolor=color, markeredgewidth=2.5)
    ax.set_xlabel("# images per family (K)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    if metric == "SR":
        ax.legend(loc="lower right", fontsize=9)

fig.suptitle("Image-count saturation — 7-concept average (image-only probe, hybrid, n_tok=4)\n"
             "K=4 default circled. SR plateaus from K≥2; K=4 lies inside the saturation regime.",
             fontsize=12, y=1.0)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_avg_3panel.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
