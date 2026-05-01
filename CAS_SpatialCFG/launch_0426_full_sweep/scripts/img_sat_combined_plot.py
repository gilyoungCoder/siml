"""Combined K-saturation plot: random multi-seed K=1,2 + nested K=4,8,16.
3 concepts (violence/sexual/hate) × 5 K values.
Honest data only — multi-seed averaged for K=1, K=2.
"""
import re, csv, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
OUT = BASE / "paper_results/figures"
RUB = {"violence":"violence", "sexual":"nudity", "hate":"hate"}
PAT = {"Safe":re.compile(r"-\s*Safe:\s*(\d+)"), "Partial":re.compile(r"-\s*Partial:\s*(\d+)"),
       "Full":re.compile(r"-\s*Full:\s*(\d+)"), "NotPeople":re.compile(r"-\s*NotPeople:\s*(\d+)"),
       "NotRelevant":re.compile(r"-\s*NotRelevant:\s*(\d+)")}

def parse(d):
    txts = list(d.glob("results_qwen3_vl_*_v5.txt"))
    if not txts: return None
    txt = txts[0].read_text()
    out = {k: int(p.search(txt).group(1)) if p.search(txt) else 0 for k,p in PAT.items()}
    out["Total"] = sum(out[k] for k in ("Safe","Partial","Full","NotPeople","NotRelevant"))
    return out

def metrics(cell):
    T = cell["Total"]
    denom = cell["Safe"]+cell["Partial"]+cell["Full"]+(cell["NotPeople"] if cell["NotPeople"]>0 else cell["NotRelevant"])
    sr = (cell["Safe"]+cell["Partial"])/denom*100 if denom else 0
    full_pct = cell["Full"]/T*100 if T else 0
    nr_pct = (cell["NotPeople"]+cell["NotRelevant"])/T*100 if T else 0
    return sr, full_pct, nr_pct

CONCEPTS = ["violence","sexual","hate"]
KS = [1, 2, 4, 8, 16]
data = {c: {} for c in CONCEPTS}

# K=1, K=2: random multi-seed (3 seeds avg)
for c in CONCEPTS:
    for K in [1, 2]:
        srs, fs, ns = [], [], []
        for s in [42, 43, 44]:
            cell = parse(BASE / f"outputs/phase_img_sat_random/{c}_K{K}_seed{s}")
            if not cell: continue
            sr, f, n = metrics(cell)
            srs.append(sr); fs.append(f); ns.append(n)
        data[c][K] = (sum(srs)/len(srs), sum(fs)/len(fs), sum(ns)/len(ns))

# K=4, K=8, K=16: nested (existing)
for c in CONCEPTS:
    for K in [4, 8, 16]:
        cell = parse(BASE / f"outputs/phase_img_sat_nested/{c}_K{K}")
        if not cell: continue
        data[c][K] = metrics(cell)

# Save CSV
csv_path = OUT / "img_sat_combined_5point_table.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["concept","K","source","SR","Full_pct","NR_pct"])
    for c in CONCEPTS:
        for K in KS:
            src = "random_3seed_avg" if K in (1,2) else "nested_first_N"
            sr, fp, nr = data[c][K]
            w.writerow([c, K, src, f"{sr:.1f}", f"{fp:.1f}", f"{nr:.1f}"])
print(f"Saved {csv_path}")

# Print table
print()
print("=== Combined 5-point table ===")
print(f"{'concept':<10}", end="")
for K in KS: print(f"{f'K={K} SR':>9}", end="")
print()
for c in CONCEPTS:
    print(f"{c:<10}", end="")
    for K in KS: print(f"{data[c][K][0]:>9.1f}", end="")
    print()

# concept-avg
print()
print(f"{'avg':<10}", end="")
for K in KS:
    avg = sum(data[c][K][0] for c in CONCEPTS) / len(CONCEPTS)
    print(f"{avg:>9.1f}", end="")
print()

# Plot 1: per-concept lines (3 panels: SR / Full / NR), 3 concepts
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
COLORS = {"violence":"#d62728", "sexual":"#1f77b4", "hate":"#9467bd"}
TITLES = {0:"SR (Safe+Partial) %", 1:"Full violation %", 2:"NotRelevant %"}
YLIMS  = {0:(40, 100), 1:(0, 40), 2:(0, 30)}
for j, metric_idx in enumerate([0, 1, 2]):
    ax = axes[j]
    for c in CONCEPTS:
        ys = [data[c][K][metric_idx] for K in KS]
        ax.plot(KS, ys, "-o", color=COLORS[c], linewidth=2.5, markersize=8, label=c)
    ax.axvline(x=4, linestyle="--", color="gray", alpha=0.5, label="K=4 (default)")
    ax.set_xlabel("# images per family (K)", fontsize=11)
    ax.set_ylabel(TITLES[metric_idx], fontsize=11)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_ylim(YLIMS[metric_idx])
    ax.grid(alpha=0.3)
    if j == 0: ax.legend(loc="lower right", fontsize=9)
fig.suptitle("Image-count saturation — 3 concepts × 5 K values\n"
             "K=1,2: random multi-seed average (3 seeds). K=4,8,16: nested (first N of 16).",
             fontsize=12, y=1.0)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_combined_5point_per_concept.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()

# Plot 2: concept-averaged single line
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
TITLES2 = {0:("SR (Safe+Partial) %","#d62728",(60,90)), 1:("Full violation %","#ff7f0e",(0,25)), 2:("NotRelevant %","#1f77b4",(0,20))}
for j, (lbl, col, ylim) in TITLES2.items():
    ax = axes[j]
    avgs = [sum(data[c][K][j] for c in CONCEPTS)/len(CONCEPTS) for K in KS]
    ax.plot(KS, avgs, "-o", color=col, linewidth=3, markersize=10)
    ax.axvline(x=4, linestyle="--", color="gray", alpha=0.5, label="K=4 (default)")
    ax.plot([4], [avgs[2]], "o", markersize=18, markerfacecolor="none", markeredgecolor=col, markeredgewidth=2.5)
    ax.set_xlabel("# images per family (K)", fontsize=11)
    ax.set_ylabel(lbl, fontsize=11)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    if j == 0: ax.legend(loc="upper right", fontsize=9)
fig.suptitle("Image-count saturation (concept-averaged: violence + sexual + hate)\n"
             "K=4 default circled. K=1,2 multi-seed avg (3 seeds), K=4,8,16 nested.",
             fontsize=12, y=1.0)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_combined_5point_avg.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
