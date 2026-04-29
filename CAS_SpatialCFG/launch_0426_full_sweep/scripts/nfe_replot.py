#!/usr/bin/env python3
"""Re-plot NFE ablation: SR + Full ratio + NotRelevant ratio panels.
Reads phase_nfe_full/<method>_<concept>_steps<N>/results_qwen3_vl_*_v5.txt
Output: nfe_curve_extended.{pdf,png} + nfe_table_extended.csv
"""
import re, csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

BASE = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
SRC  = BASE / "outputs/phase_nfe_full"
OUT  = BASE / "paper_results/figures"
OUT.mkdir(parents=True, exist_ok=True)

CONCEPTS = ["violence", "shocking", "self-harm", "sexual"]
METHODS  = ["ebsg", "safree", "baseline"]
STEPS    = [1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]

CONCEPT_RUBRIC = {"violence":"violence", "shocking":"shocking", "self-harm":"self_harm", "sexual":"nudity"}

# regex on results_qwen3_vl_<rubric>_v5.txt
PAT = {
    "Safe":       re.compile(r"-\s*Safe:\s*(\d+)\s*\((?P<pct>[\d.]+)%\)"),
    "Partial":    re.compile(r"-\s*Partial:\s*(\d+)"),
    "Full":       re.compile(r"-\s*Full:\s*(\d+)"),
    "NotPeople":  re.compile(r"-\s*NotPeople:\s*(\d+)"),
    "NotRelevant":re.compile(r"-\s*NotRelevant:\s*(\d+)"),
}

def parse_cell(d: Path):
    """Return dict with counts for S/P/F/NP/NR + total."""
    txts = list(d.glob("results_qwen3_vl_*_v5.txt"))
    if not txts: return None
    txt = txts[0].read_text()
    out = {"Safe":0, "Partial":0, "Full":0, "NotPeople":0, "NotRelevant":0}
    for k, pat in PAT.items():
        m = pat.search(txt)
        if m: out[k] = int(m.group(1))
    out["Total"] = out["Safe"] + out["Partial"] + out["Full"] + out["NotPeople"] + out["NotRelevant"]
    return out

# Collect data
rows = []  # (concept, method, steps, S, P, F, NP, NR, Total, SR, FullPct, NRPct)
for c in CONCEPTS:
    for m in METHODS:
        for n in STEPS:
            d = SRC / f"{m}_{c}_steps{n}"
            cell = parse_cell(d)
            if cell is None or cell["Total"] == 0:
                rows.append((c, m, n, 0, 0, 0, 0, 0, 0, None, None, None))
                continue
            T = cell["Total"]
            denom = cell["Safe"] + cell["Partial"] + cell["Full"] + (cell["NotPeople"] if cell["NotPeople"]>0 else cell["NotRelevant"])
            sr = (cell["Safe"] + cell["Partial"]) / denom * 100 if denom else None
            full_pct = cell["Full"] / T * 100 if T else None
            nr_pct = (cell["NotRelevant"] + cell["NotPeople"]) / T * 100 if T else None
            rows.append((c, m, n, cell["Safe"], cell["Partial"], cell["Full"],
                         cell["NotPeople"], cell["NotRelevant"], T, sr, full_pct, nr_pct))

# Save extended CSV
csv_path = OUT / "nfe_table_extended.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["concept","method","steps","Safe","Partial","Full","NotPeople","NotRelevant","Total","SR_pct","Full_pct","NR_pct"])
    for r in rows:
        w.writerow(r)
print(f"Saved {csv_path}")

# Build nested dict for plotting
def get(c, m, metric):
    """metric in {SR_pct, Full_pct, NR_pct} (col idx 9, 10, 11)"""
    idx = {"SR":9, "Full":10, "NR":11}[metric]
    out = []
    for r in rows:
        if r[0]==c and r[1]==m:
            out.append((r[2], r[idx]))
    out.sort()
    return [s for s,v in out], [v for s,v in out]

# Plot 4×3 grid: rows=concepts, cols=metrics(SR, Full, NR)
fig, axes = plt.subplots(len(CONCEPTS), 3, figsize=(13, 11), sharex=True)
COLORS = {"ebsg":"#d62728", "safree":"#1f77b4", "baseline":"#7f7f7f"}
LABELS = {"ebsg":"EBSG (ours)", "safree":"SAFREE", "baseline":"SD1.4 baseline"}
TITLES = {"SR":"SR (Safe+Partial) %", "Full":"Full violation %", "NR":"NotRelevant %"}

for i, c in enumerate(CONCEPTS):
    for j, metric in enumerate(["SR","Full","NR"]):
        ax = axes[i, j]
        for m in METHODS:
            xs, ys = get(c, m, metric)
            xs = [x for x,y in zip(xs,ys) if y is not None]
            ys = [y for y in ys if y is not None]
            ax.plot(xs, ys, "-o", color=COLORS[m], label=LABELS[m], linewidth=2, markersize=4)
        if j == 0:
            ax.set_ylabel(c, fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_title(TITLES[metric], fontsize=11)
        if i == len(CONCEPTS)-1:
            ax.set_xlabel("Denoising steps (NFE)", fontsize=10)
        ax.grid(alpha=0.3)
        # ax.set_xscale removed (linear)
        ax.set_xticks([1, 5, 10, 20, 30, 50])
        ax.set_xticklabels(["1","5","10","20","30","50"])
        ax.set_ylim(-5, 105)
        if i == 0 and j == 0:
            ax.legend(loc="lower right", fontsize=8)

fig.suptitle("NFE Ablation — SR / Full violation / NotRelevant per (concept, method, step)",
             fontsize=13, y=0.995)
fig.tight_layout()
for ext in ("png","pdf"):
    p = OUT / f"nfe_curve_extended.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
