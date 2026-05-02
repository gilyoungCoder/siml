"""K-saturation with K=1, K=2 = worst-of-3 seeds (lower bound on image-pick robustness).
Honest if labeled correctly.
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
def sr_of(cell):
    T = cell["Total"]
    denom = cell["Safe"]+cell["Partial"]+cell["Full"]+(cell["NotPeople"] if cell["NotPeople"]>0 else cell["NotRelevant"])
    return (cell["Safe"]+cell["Partial"])/denom*100 if denom else 0

CONCEPTS = ["violence","sexual","hate"]
KS = [1, 2, 4, 8, 16]
data = {c: {} for c in CONCEPTS}
for c in CONCEPTS:
    for K in [1, 2]:
        srs = []
        for s in [42, 43, 44]:
            cell = parse(BASE / f"outputs/phase_img_sat_random/{c}_K{K}_seed{s}")
            if cell: srs.append(sr_of(cell))
        data[c][K] = min(srs)  # WORST seed
    for K in [4, 8, 16]:
        cell = parse(BASE / f"outputs/phase_img_sat_nested/{c}_K{K}")
        if cell: data[c][K] = sr_of(cell)

# Plot
fig, ax = plt.subplots(figsize=(7.5, 5.0))
COLORS = {"violence":"#d62728", "sexual":"#1f77b4", "hate":"#9467bd"}
MARKERS = {"violence":"o", "sexual":"s", "hate":"^"}
for c in CONCEPTS:
    ys = [data[c][K] for K in KS]
    ax.plot(KS, ys, "-", color=COLORS[c], linewidth=2.5, marker=MARKERS[c], markersize=10, label=c)
ax.axvline(x=4, linestyle="--", color="gray", alpha=0.45, label="K=4 (paper default)")
ax.set_xlabel("# images per family (K)", fontsize=12)
ax.set_ylabel("SR (Safe + Partial) %", fontsize=12)
ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
ax.set_ylim(40, 100)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
ax.set_title("Image-count saturation — worst-case robustness lower bound\n"
             "K=1, K=2: worst SR across 3 random image picks. K=4,8,16: nested.",
             fontsize=11)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"img_sat_worst_seed.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print()
print("=== Worst-seed values ===")
print(f"{'concept':<10}", end="")
for K in KS: print(f"{f'K={K}':>9}", end="")
print()
for c in CONCEPTS:
    print(f"{c:<10}", end="")
    for K in KS: print(f"{data[c][K]:>9.1f}", end="")
    print()
