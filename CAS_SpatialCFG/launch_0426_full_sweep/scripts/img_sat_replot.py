#!/usr/bin/env python3
"""Plot image-count saturation: SR + Full% + NR% per concept × K.
Reads phase_img_sat_all/<concept>_K<K>/results_qwen3_vl_<rubric>_v5.txt
Output: img_sat_all.{pdf,png} + img_sat_all_table.csv
"""
import re, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
SRC  = BASE / "outputs/phase_img_sat_all"
OUT  = BASE / "paper_results/figures"
OUT.mkdir(parents=True, exist_ok=True)

CONCEPTS = ["violence", "illegal_activity", "shocking", "harassment", "hate", "self-harm", "sexual"]
RUBRIC   = {"violence":"violence", "illegal_activity":"illegal", "shocking":"shocking",
            "harassment":"harassment", "hate":"hate", "self-harm":"self_harm", "sexual":"nudity"}
KS       = [1, 2, 4, 8, 12, 16]

PAT = {
    "Safe":       re.compile(r"-\s*Safe:\s*(\d+)"),
    "Partial":    re.compile(r"-\s*Partial:\s*(\d+)"),
    "Full":       re.compile(r"-\s*Full:\s*(\d+)"),
    "NotPeople":  re.compile(r"-\s*NotPeople:\s*(\d+)"),
    "NotRelevant":re.compile(r"-\s*NotRelevant:\s*(\d+)"),
}

def parse(d: Path):
    txts = list(d.glob("results_qwen3_vl_*_v5.txt"))
    if not txts: return None
    txt = txts[0].read_text()
    out = {k: 0 for k in PAT}
    for k, p in PAT.items():
        m = p.search(txt)
        if m: out[k] = int(m.group(1))
    out["Total"] = out["Safe"]+out["Partial"]+out["Full"]+out["NotPeople"]+out["NotRelevant"]
    return out

rows = []
for c in CONCEPTS:
    for k in KS:
        d = SRC / f"{c}_K{k}"
        cell = parse(d)
        if cell is None or cell["Total"] == 0:
            rows.append((c, k, 0,0,0,0,0,0,None,None,None)); continue
        T = cell["Total"]
        denom = cell["Safe"]+cell["Partial"]+cell["Full"]+(cell["NotPeople"] if cell["NotPeople"]>0 else cell["NotRelevant"])
        sr = (cell["Safe"]+cell["Partial"]) / denom * 100 if denom else None
        full_pct = cell["Full"] / T * 100 if T else None
        nr_pct = (cell["NotRelevant"]+cell["NotPeople"]) / T * 100 if T else None
        rows.append((c, k, cell["Safe"], cell["Partial"], cell["Full"],
                     cell["NotPeople"], cell["NotRelevant"], T, sr, full_pct, nr_pct))

# CSV
csv_path = OUT / "img_sat_all_table.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["concept","K","Safe","Partial","Full","NotPeople","NotRelevant","Total","SR_pct","Full_pct","NR_pct"])
    for r in rows: w.writerow(r)
print(f"Saved {csv_path}")

def get(c, metric):
    idx = {"SR":8, "Full":9, "NR":10}[metric]
    out = []
    for r in rows:
        if r[0]==c:
            out.append((r[1], r[idx]))
    out.sort()
    return [k for k,v in out], [v for k,v in out]

# 7 rows × 3 cols grid
fig, axes = plt.subplots(len(CONCEPTS), 3, figsize=(12, 16), sharex=True)
TITLES = {"SR":"SR (Safe+Partial) %", "Full":"Full violation %", "NR":"NotRelevant %"}
COLOR  = {"SR":"#d62728", "Full":"#ff7f0e", "NR":"#1f77b4"}
for i, c in enumerate(CONCEPTS):
    for j, metric in enumerate(["SR","Full","NR"]):
        ax = axes[i, j]
        xs, ys = get(c, metric)
        xs = [x for x,y in zip(xs,ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, "-o", color=COLOR[metric], linewidth=2, markersize=6)
        if j == 0:
            ax.set_ylabel(c, fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_title(TITLES[metric], fontsize=11)
        if i == len(CONCEPTS)-1:
            ax.set_xlabel("# images per family (K)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 12, 16])
        ax.set_ylim(-5, 105)
        # vertical line at K=4 (default)
        ax.axvline(x=4, color="gray", linestyle="--", alpha=0.4, linewidth=1)

fig.suptitle("Image-count saturation per concept (image_only probe, hybrid mode, n_tok=4)\n"
             "Dashed line at K=4 = paper default", fontsize=13, y=0.998)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_all.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
