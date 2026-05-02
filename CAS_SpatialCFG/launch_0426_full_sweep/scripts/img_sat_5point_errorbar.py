"""5-point K-saturation with error bars (min/mean/max) for K=1, K=2.
3 concepts × 5 K. Honest data, no cherry-picking.
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
OUT = BASE / "paper_results/figures"
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
    return ((cell["Safe"]+cell["Partial"])/denom*100 if denom else 0,
            cell["Full"]/T*100 if T else 0,
            (cell["NotPeople"]+cell["NotRelevant"])/T*100 if T else 0)

CONCEPTS = ["violence","sexual","hate"]
KS = [1, 2, 4, 8, 16]

# K=1, K=2 multi-seed; K=4,8,16 nested single
data = {c: {} for c in CONCEPTS}
for c in CONCEPTS:
    for K in [1, 2]:
        srs = []
        for s in [42, 43, 44]:
            cell = parse(BASE / f"outputs/phase_img_sat_random/{c}_K{K}_seed{s}")
            if cell: srs.append(metrics(cell)[0])
        data[c][K] = (np.mean(srs), np.min(srs), np.max(srs))
    for K in [4, 8, 16]:
        cell = parse(BASE / f"outputs/phase_img_sat_nested/{c}_K{K}")
        if cell:
            sr = metrics(cell)[0]
            data[c][K] = (sr, sr, sr)

# Plot: per-concept SR with error bars
fig, ax = plt.subplots(figsize=(7.5, 5.2))
COLORS = {"violence":"#d62728", "sexual":"#1f77b4", "hate":"#9467bd"}
MARKERS = {"violence":"o", "sexual":"s", "hate":"^"}
for c in CONCEPTS:
    means = [data[c][K][0] for K in KS]
    mins  = [data[c][K][1] for K in KS]
    maxs  = [data[c][K][2] for K in KS]
    yerr_low  = [m - lo for m, lo in zip(means, mins)]
    yerr_high = [hi - m for hi, m in zip(maxs, means)]
    ax.errorbar(KS, means, yerr=[yerr_low, yerr_high], marker=MARKERS[c], color=COLORS[c],
                linewidth=2.2, markersize=9, capsize=5, capthick=2, label=c)
ax.axvline(x=4, linestyle="--", color="gray", alpha=0.45, label="K=4 (paper default)")
ax.set_xlabel("# images per family (K)", fontsize=12)
ax.set_ylabel("SR (Safe + Partial) %", fontsize=12)
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.set_ylim(40, 100)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
ax.set_title("Image-count saturation (3 concepts × 5 K values)\n"
             "K=1, K=2: 3-seed (random pick) — error bar = min/max range. K=4,8,16: nested.",
             fontsize=11)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_5point_errorbar.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()

# Concept-averaged version
fig, ax = plt.subplots(figsize=(7.5, 5.2))
all_means_per_K = []
all_mins_per_K  = []
all_maxs_per_K  = []
for K in KS:
    avg_mean = np.mean([data[c][K][0] for c in CONCEPTS])
    avg_min  = np.mean([data[c][K][1] for c in CONCEPTS])  # avg of min across concepts
    avg_max  = np.mean([data[c][K][2] for c in CONCEPTS])
    all_means_per_K.append(avg_mean); all_mins_per_K.append(avg_min); all_maxs_per_K.append(avg_max)

yerr_low  = [m - lo for m, lo in zip(all_means_per_K, all_mins_per_K)]
yerr_high = [hi - m for hi, m in zip(all_maxs_per_K, all_means_per_K)]
ax.errorbar(KS, all_means_per_K, yerr=[yerr_low, yerr_high], marker="o", color="#d62728",
            linewidth=3, markersize=11, capsize=6, capthick=2.5)
ax.axvline(x=4, linestyle="--", color="gray", alpha=0.45, label="K=4 (paper default)")
ax.plot([4], [all_means_per_K[2]], "o", markersize=20, markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=2.5)
ax.set_xlabel("# images per family (K)", fontsize=12)
ax.set_ylabel("SR (Safe + Partial) % — concept avg", fontsize=12)
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.set_ylim(60, 90)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
ax.set_title("Image-count saturation — averaged across violence + sexual + hate\n"
             "Error bars = average min/max across 3 concepts (K=1,2 multi-seed).",
             fontsize=11)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_5point_avg_errorbar.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")

print()
print("=== Final 5-point table (mean, with min/max for K=1,2) ===")
print(f"{'concept':<10}", end="")
for K in KS: print(f"{f'K={K}':>14}", end="")
print()
for c in CONCEPTS:
    print(f"{c:<10}", end="")
    for K in KS:
        m, lo, hi = data[c][K]
        if lo == hi:
            print(f"{m:>10.1f}     ", end="")
        else:
            print(f"{m:>5.1f}[{lo:.0f}-{hi:.0f}]", end="")
    print()
