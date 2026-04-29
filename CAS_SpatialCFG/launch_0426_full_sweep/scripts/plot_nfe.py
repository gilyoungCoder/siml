#!/usr/bin/env python3
"""Plot NFE ablation: SR vs steps, 4 subplots (one per concept), 3 lines (EBSG/SAFREE/Baseline)."""
import json, os, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_full"
OUTDIR = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

CONCEPTS = ["violence", "shocking", "self-harm", "sexual"]
STEPS = [1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
METHODS = ["ebsg", "safree", "baseline"]
RUBRIC = {"violence":"violence","shocking":"shocking","self-harm":"self_harm","sexual":"nudity"}
COLOR = {"ebsg":"#d62728", "safree":"#1f77b4", "baseline":"#7f7f7f"}
LABEL = {"ebsg":"EBSG (Ours)", "safree":"SAFREE", "baseline":"SD1.4 baseline"}
MARKER = {"ebsg":"o", "safree":"s", "baseline":"^"}

def sr(cell_dir, rubric):
    js = f"{cell_dir}/categories_qwen3_vl_{rubric}_v5.json"
    if not os.path.isfile(js): return None
    counts = {}
    for v in json.load(open(js)).values():
        lbl = v.get("category","?") if isinstance(v,dict) else str(v)
        counts[lbl] = counts.get(lbl,0)+1
    s,p,f = counts.get("Safe",0), counts.get("Partial",0), counts.get("Full",0)
    np_, nr = counts.get("NotPeople",0), counts.get("NotRelevant",0)
    den = s+p+f+(np_ if np_>0 else nr)
    return 100.0*(s+p)/den if den>0 else None

# Build data: data[concept][method] = list of (step, sr) sorted
data = {}
for concept in CONCEPTS:
    data[concept] = {}
    rubric = RUBRIC[concept]
    for method in METHODS:
        ys = []
        for step in STEPS:
            cell = f"{BASE}/{method}_{concept}_steps{step}"
            v = sr(cell, rubric)
            ys.append(v)
        data[concept][method] = ys

# Plot 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
axes = axes.flatten()
for ax, concept in zip(axes, CONCEPTS):
    for method in METHODS:
        ys = data[concept][method]
        # Filter Nones for plot
        xs_valid = [s for s, y in zip(STEPS, ys) if y is not None]
        ys_valid = [y for y in ys if y is not None]
        ax.plot(xs_valid, ys_valid, color=COLOR[method], marker=MARKER[method],
                markersize=6, linewidth=2.0, label=LABEL[method])
    ax.set_title(f"{concept}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of denoising steps (NFE)", fontsize=11)
    ax.set_ylabel("Safety Rate (%)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 5, 10, 20, 30, 50])
    ax.set_ylim(-3, 100)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

fig.suptitle("Safety Rate vs. Denoising Steps (NFE) — EBSG vs SAFREE vs Baseline", fontsize=14)
plt.tight_layout()
out_pdf = f"{OUTDIR}/nfe_curve.pdf"
out_png = f"{OUTDIR}/nfe_curve.png"
plt.savefig(out_pdf, dpi=150)
plt.savefig(out_png, dpi=150)
print(f"saved: {out_pdf}\nsaved: {out_png}")

# Also dump CSV table
csv_path = f"{OUTDIR}/nfe_table.csv"
with open(csv_path, "w") as f:
    f.write("concept,method," + ",".join(f"steps={s}" for s in STEPS) + "\n")
    for concept in CONCEPTS:
        for method in METHODS:
            ys = data[concept][method]
            f.write(f"{concept},{method}," + ",".join(f"{y:.2f}" if y is not None else "—" for y in ys) + "\n")
print(f"saved: {csv_path}")
