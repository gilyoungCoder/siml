#!/usr/bin/env python3
"""5-method NFE comparison + per-image timing chart.
Output: nfe_curve_5method.{pdf,png} + nfe_timing_5method.{pdf,png} + nfe_5method_table.csv
Methods: ebsg, safree, baseline (existing) + safedenoiser, sgf (new).
"""
import re, csv, glob, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
SRC1 = BASE / "outputs/phase_nfe_full"
SRC2 = BASE / "outputs/phase_nfe_safedenoiser_sgf"
OUT  = BASE / "paper_results/figures"
OUT.mkdir(parents=True, exist_ok=True)

CONCEPTS = ["violence", "shocking", "self-harm", "sexual"]
METHODS  = ["ebsg", "safree", "baseline", "safedenoiser", "sgf"]
STEPS    = [1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
RUBRIC   = {"violence":"violence", "shocking":"shocking", "self-harm":"self_harm", "sexual":"nudity"}

PAT = {
    "Safe":       re.compile(r"-\s*Safe:\s*(\d+)"),
    "Partial":    re.compile(r"-\s*Partial:\s*(\d+)"),
    "Full":       re.compile(r"-\s*Full:\s*(\d+)"),
    "NotPeople":  re.compile(r"-\s*NotPeople:\s*(\d+)"),
    "NotRelevant":re.compile(r"-\s*NotRelevant:\s*(\d+)"),
}

def parse_results_txt(path):
    if not os.path.isfile(path): return None
    txt = open(path).read()
    out = {k: 0 for k in PAT}
    for k, p in PAT.items():
        m = p.search(txt)
        if m: out[k] = int(m.group(1))
    out["Total"] = sum(out[k] for k in ("Safe","Partial","Full","NotPeople","NotRelevant"))
    return out

def cell_dir(method, concept, step):
    if method in ("ebsg","safree","baseline"):
        return SRC1 / f"{method}_{concept}_steps{step}"
    return SRC2 / f"{method}_{concept}_step{step}/all"

def cell_results_path(method, concept, step):
    rubric = RUBRIC[concept]
    return cell_dir(method, concept, step) / f"results_qwen3_vl_{rubric}_v5.txt"

# Build master table
rows = []
for m in METHODS:
    for c in CONCEPTS:
        for s in STEPS:
            cell = parse_results_txt(str(cell_results_path(m, c, s)))
            if cell is None or cell["Total"] == 0:
                rows.append((m, c, s, 0,0,0,0,0,0, None, None, None)); continue
            T = cell["Total"]
            denom = cell["Safe"]+cell["Partial"]+cell["Full"]+(cell["NotPeople"] if cell["NotPeople"]>0 else cell["NotRelevant"])
            sr = (cell["Safe"]+cell["Partial"]) / denom * 100 if denom else None
            full_pct = cell["Full"] / T * 100 if T else None
            nr_pct = (cell["NotRelevant"]+cell["NotPeople"]) / T * 100 if T else None
            rows.append((m, c, s, cell["Safe"], cell["Partial"], cell["Full"],
                         cell["NotPeople"], cell["NotRelevant"], T, sr, full_pct, nr_pct))

csv_path = OUT / "nfe_5method_table.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","concept","step","Safe","Partial","Full","NotPeople","NotRelevant","Total","SR_pct","Full_pct","NR_pct"])
    for r in rows: w.writerow(r)
print(f"Saved {csv_path}")

# Plot 1: SR + Full% + NR% per concept × method (4 concept rows × 3 metric cols)
def get(method, concept, metric):
    idx = {"SR":9, "Full":10, "NR":11}[metric]
    pts = []
    for r in rows:
        if r[0]==method and r[1]==concept and r[idx] is not None:
            pts.append((r[2], r[idx]))
    pts.sort()
    return [s for s,v in pts], [v for s,v in pts]

COLORS = {"ebsg":"#d62728", "safree":"#1f77b4", "baseline":"#7f7f7f",
          "safedenoiser":"#2ca02c", "sgf":"#9467bd"}
LABELS = {"ebsg":"EBSG (ours)", "safree":"SAFREE", "baseline":"SD1.4 baseline",
          "safedenoiser":"SafeDenoiser", "sgf":"SGF"}
TITLES = {"SR":"SR (Safe+Partial) %", "Full":"Full violation %", "NR":"NotRelevant %"}

fig, axes = plt.subplots(len(CONCEPTS), 3, figsize=(13, 11), sharex=True)
for i, c in enumerate(CONCEPTS):
    for j, metric in enumerate(["SR","Full","NR"]):
        ax = axes[i, j]
        for m in METHODS:
            xs, ys = get(m, c, metric)
            if not xs: continue
            ax.plot(xs, ys, "-o", color=COLORS[m], label=LABELS[m], linewidth=2, markersize=4)
        if j == 0: ax.set_ylabel(c, fontsize=11, fontweight="bold")
        if i == 0: ax.set_title(TITLES[metric], fontsize=11)
        if i == len(CONCEPTS)-1: ax.set_xlabel("Denoising steps (NFE)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks([1, 5, 10, 20, 30, 50])
        ax.set_ylim(-5, 105)
        if i == 0 and j == 0:
            ax.legend(loc="lower right", fontsize=8)
fig.suptitle("NFE Ablation — 5-method comparison (SR / Full violation / NotRelevant)", fontsize=13, y=0.995)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"nfe_curve_5method.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()

# Plot 2: Timing — per-image gen time vs NFE (averaged across 4 concepts)
import collections
TIMING_CSV = OUT / "nfe_5method_timing.csv"
timing = collections.defaultdict(list)
with open(TIMING_CSV) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        if r["per_img_sec"] in ("", "None"): continue
        timing[(r["method"], int(r["step"]))].append(float(r["per_img_sec"]))
avg_t = {k: sum(v)/len(v) for k,v in timing.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for m in METHODS:
    xs, ys = [], []
    for s in STEPS:
        if (m, s) in avg_t:
            xs.append(s); ys.append(avg_t[(m, s)])
    ax.plot(xs, ys, "-o", color=COLORS[m], label=LABELS[m], linewidth=2, markersize=5)
ax.set_xlabel("Denoising steps (NFE)", fontsize=11)
ax.set_ylabel("Per-image generation time (s)", fontsize=11)
ax.set_title("Inference cost vs NFE (avg over 4 concepts)", fontsize=12)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
# per-step time = per_img / step
for m in METHODS:
    xs, ys = [], []
    for s in STEPS:
        if (m, s) in avg_t:
            xs.append(s); ys.append(avg_t[(m, s)] / s)
    ax.plot(xs, ys, "-o", color=COLORS[m], label=LABELS[m], linewidth=2, markersize=5)
ax.set_xlabel("Denoising steps (NFE)", fontsize=11)
ax.set_ylabel("Per-step time (s/step)", fontsize=11)
ax.set_title("Per-step UNet forward cost (overhead view)", fontsize=12)
ax.set_xscale("log"); ax.set_xticks([1,5,10,20,50]); ax.set_xticklabels(["1","5","10","20","50"])
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"nfe_timing_5method.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
