#!/usr/bin/env python3
"""Plot image-count saturation: SR vs K (imgs per family)."""
import json, os, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_img_saturation"
OUTDIR = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 2, 4, 8, 16, 32]
RUBRIC = "nudity"

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

ys = []
for K in K_VALUES:
    cell = f"{BASE}/sexual_K{K}"
    v = sr(cell, RUBRIC)
    ys.append(v)
    print(f"K={K}: SR={v}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(K_VALUES, ys, color="#d62728", marker="o", markersize=10, linewidth=2.5, label="EBSG (sexual)")
ax.set_xlabel("Number of exemplars per family (K)", fontsize=13)
ax.set_ylabel("Safety Rate (%)", fontsize=13)
ax.set_title("Image-count saturation: SR vs K (sexual concept)", fontsize=14, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(K_VALUES)
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.grid(alpha=0.3, which="both")
ax.set_ylim(50, 100)
ax.legend(loc="lower right", fontsize=11)
for x, y in zip(K_VALUES, ys):
    if y is not None:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,10),
                    ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/img_saturation.pdf", dpi=150)
plt.savefig(f"{OUTDIR}/img_saturation.png", dpi=150)
print(f"saved: {OUTDIR}/img_saturation.{{pdf,png}}")

# CSV
with open(f"{OUTDIR}/img_saturation.csv", "w") as f:
    f.write("K,SR\n")
    for k, y in zip(K_VALUES, ys):
        f.write(f"{k},{y if y is not None else ''}\n")
