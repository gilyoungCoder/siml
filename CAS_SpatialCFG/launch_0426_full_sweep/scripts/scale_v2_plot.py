"""Scale-robustness v2 plot: 4 SLD variants vs EBSG, sexual concept, gs sweep [5..500].
Includes EBSG data from previous scale_robustness CSV (matched ss ∈ [5,500]).
"""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
DATA_V2 = ROOT / "outputs/phase_scale_robustness_v2"
DATA_V1 = ROOT / "outputs/phase_scale_robustness"  # for EBSG sexual
OUT_DIR = ROOT / "paper_results/figures"

CONCEPT = "sexual"
RUBRIC = "nudity"
SCALES = [5, 10, 20, 50, 100, 200, 500]
VARIANTS = ["max", "strong", "medium", "weak"]
PAT = re.compile(r"-\s*(Safe|Partial|Full|NotPeople|NotRelevant):\s*(\d+)\s*\(([-0-9.]+)%\)")


def parse_result(p):
    if p is None or not p.exists() or p.stat().st_size < 50: return None
    txt = p.read_text(errors="ignore")
    counts = {"Safe":0,"Partial":0,"Full":0,"NotPeople":0,"NotRelevant":0}
    for m in PAT.finditer(txt): counts[m.group(1)] = int(m.group(2))
    S,P,F,NP,NR = counts["Safe"],counts["Partial"],counts["Full"],counts["NotPeople"],counts["NotRelevant"]
    T = S+P+F+NP+NR
    if T == 0: return None
    denom = S+P+F + (NP if NP > 0 else NR)
    sr = (S+P)/denom*100 if denom else 0
    return sr, F/T*100, (NP+NR)/T*100


# Parse SLD variants
data = {}
for v in VARIANTS:
    for s in SCALES:
        cell = DATA_V2 / f"sld_{v}_{CONCEPT}_scale{s}"
        f = cell / f"results_qwen3_vl_{RUBRIC}_v5.txt"
        r = parse_result(f)
        if r: data[(f"sld_{v}", s)] = r

# Parse EBSG from v1 (sexual)
for s in SCALES:
    cell = DATA_V1 / f"ebsg_{CONCEPT}_scale{s}"
    f = cell / f"results_qwen3_vl_{RUBRIC}_v5.txt"
    r = parse_result(f)
    if r: data[("ebsg", s)] = r

# CSV dump
csv_out = OUT_DIR / "scale_robustness_v2_table.csv"
with csv_out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "scale", "SR_pct", "Full_pct", "NR_pct"])
    for (m, s), (sr, fp, nr) in sorted(data.items(), key=lambda x: (x[0][0], x[0][1])):
        w.writerow([m, s, f"{sr:.1f}", f"{fp:.1f}", f"{nr:.1f}"])
print(f"Saved {csv_out}  (rows={len(data)})")

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

STYLE = {
    "sld_max":    dict(color="#5a009e", linestyle="--", linewidth=1.8, marker="P", markersize=8, alpha=0.95),
    "sld_strong": dict(color="#8a2be2", linestyle="--", linewidth=1.6, marker="X", markersize=8, alpha=0.9),
    "sld_medium": dict(color="#b07ad9", linestyle=":",  linewidth=1.6, marker="h", markersize=8, alpha=0.9),
    "sld_weak":   dict(color="#d8b3ff", linestyle=":",  linewidth=1.4, marker="p", markersize=8, alpha=0.85),
    "ebsg":       dict(color="#c11c1c", linestyle="-",  linewidth=2.6, marker="*", markersize=14, alpha=1.0, zorder=5),
}
LABELS = {
    "sld_max":    "SLD-Max (clamp off)",
    "sld_strong": "SLD-Strong (clamp off)",
    "sld_medium": "SLD-Medium (clamp off)",
    "sld_weak":   "SLD-Weak (clamp off)",
    "ebsg":       "EBSG (Ours)",
}
TITLES = ["SR (Safe + Partial) %  ↑", "Full violation %  ↓", "NotRelevant %  ↓"]
YLIMS = [(0, 100), (0, 80), (0, 100)]

# 1 row × 3 col
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
methods = ["sld_max", "sld_strong", "sld_medium", "sld_weak", "ebsg"]
for j, (title, ylim) in enumerate(zip(TITLES, YLIMS)):
    ax = axes[j]
    ax.set_xscale("log")
    ax.set_xlabel("Guidance scale")
    ax.set_ylabel(title); ax.set_ylim(ylim)
    ax.grid(which="both", alpha=0.25, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for m in methods:
        xs = [s for s in SCALES if (m, s) in data]
        ys = [data[(m, s)][j] for s in xs]
        if xs: ax.plot(xs, ys, label=LABELS[m], **STYLE[m])
    if j == 0:
        ax.legend(loc="best", framealpha=0.95, edgecolor="0.7", fontsize=8)

fig.suptitle(f"Scale robustness, sexual concept (NFE=50, 60 prompts): SLD 4 variants vs EBSG.\n"
             f"All sweeps over guidance/safety scale ∈ {SCALES}. SLD clamp removed (SLD_CLAMP_MAX=1e6).",
             fontsize=10.5, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    p = OUT_DIR / f"scale_robustness_v2.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
