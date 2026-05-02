"""Scale-robustness plot: SR / Full / NotRel vs scale, comparing SLD and EBSG.
Shows: EBSG stays robust at very high safety_scale (when/where mechanism prevents
collapse), while SLD's NotRel surges as sld_guidance_scale increases.
"""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
DATA_DIR = ROOT / "outputs/phase_scale_robustness"
OUT_DIR  = ROOT / "paper_results/figures"

CONCEPTS = ["sexual", "violence"]
SCALES_SLD  = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
SCALES_EBSG = [5, 10, 20, 50, 100, 200, 500]

EVAL_NAME = {"sexual": "nudity", "violence": "violence"}
PAT = re.compile(r"-\s*(Safe|Partial|Full|NotPeople|NotRelevant):\s*(\d+)\s*\(([-0-9.]+)%\)")


def parse_result(p):
    if not p.exists() or p.stat().st_size < 50: return None
    txt = p.read_text(errors="ignore")
    counts = {"Safe": 0, "Partial": 0, "Full": 0, "NotPeople": 0, "NotRelevant": 0}
    for m in PAT.finditer(txt):
        counts[m.group(1)] = int(m.group(2))
    S, P, F, NP, NR = counts["Safe"], counts["Partial"], counts["Full"], counts["NotPeople"], counts["NotRelevant"]
    T = S + P + F + NP + NR
    if T == 0: return None
    denom = S + P + F + (NP if NP > 0 else NR)
    sr = (S + P) / denom * 100 if denom else 0
    return sr, F / T * 100, (NP + NR) / T * 100


def find_result(method, c, s):
    cell = DATA_DIR / f"{method}_{c}_scale{s}"
    rub = EVAL_NAME[c]
    f = cell / f"results_qwen3_vl_{rub}_v5.txt"
    return f if f.exists() and f.stat().st_size > 50 else None


# Parse
data = {}  # (method, concept, scale) -> (SR, Full, NR)
for c in CONCEPTS:
    for s in SCALES_SLD:
        r = parse_result(find_result("sld", c, s) or Path("/dev/null"))
        if r: data[("sld", c, s)] = r
    for s in SCALES_EBSG:
        r = parse_result(find_result("ebsg", c, s) or Path("/dev/null"))
        if r: data[("ebsg", c, s)] = r

# CSV dump
csv_out = OUT_DIR / "scale_robustness_table.csv"
with csv_out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "concept", "scale", "SR_pct", "Full_pct", "NR_pct"])
    for (m, c, s), (sr, fp, nr) in sorted(data.items()):
        w.writerow([m, c, s, f"{sr:.1f}", f"{fp:.1f}", f"{nr:.1f}"])
print(f"Saved {csv_out}  (rows={len(data)})")

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

# Plot: 2 row (concept) × 3 col (SR/Full/NR), TWIN x-axes overlaid for SLD vs EBSG
COL_TITLES = ["SR (Safe + Partial) %  ↑", "Full violation %  ↓", "NotRelevant %  ↓"]
YLIMS = [(0, 100), (0, 80), (0, 100)]
SLD_STYLE  = dict(color="#8a2be2", linestyle="--", linewidth=2.0, marker="P", markersize=8, alpha=0.95)
EBSG_STYLE = dict(color="#c11c1c", linestyle="-",  linewidth=2.6, marker="*", markersize=12, alpha=1.0, zorder=5)

fig, axes = plt.subplots(len(CONCEPTS), 3, figsize=(13.5, 3.4 * len(CONCEPTS)))
for ci, c in enumerate(CONCEPTS):
    for j, (title, ylim) in enumerate(zip(COL_TITLES, YLIMS)):
        ax = axes[ci, j]
        ax.set_xscale("log")
        ax.set_xlabel("safety / SLD guidance scale  (log)", fontsize=10)
        ax.set_ylabel(f"{c}\n{title}" if j == 0 else title, fontsize=10)
        ax.set_ylim(ylim)
        ax.grid(which="both", alpha=0.25, linestyle="-", linewidth=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # SLD
        xs = [s for s in SCALES_SLD if ("sld", c, s) in data]
        ys = [data[("sld", c, s)][j] for s in xs]
        if xs:
            ax.plot(xs, ys, label="SLD-Max (sweep gs)", **SLD_STYLE)
        # EBSG (uses different absolute scale; both go on same log x)
        xs = [s for s in SCALES_EBSG if ("ebsg", c, s) in data]
        ys = [data[("ebsg", c, s)][j] for s in xs]
        if xs:
            ax.plot(xs, ys, label="EBSG (sweep ss)", **EBSG_STYLE)
        if ci == 0 and j == 0:
            ax.legend(loc="best", framealpha=0.95, edgecolor="0.7")

fig.suptitle("Robustness to safety-scale: EBSG stays stable at extreme scales while SLD-Max degenerates "
             "(NotRel surge)\n"
             "X-axis: log10 of safety/SLD guidance scale. EBSG sweeps safety_scale ∈ {5,10,20,50,100,200,500}; "
             "SLD-Max sweeps sld_guidance_scale ∈ {1000..100000}.",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    p = OUT_DIR / f"scale_robustness.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()


# NeurIPS-sized headline (single concept-avg compact)
def _avg(metric_idx):
    rows = []
    for m, scales in [("sld", SCALES_SLD), ("ebsg", SCALES_EBSG)]:
        for s in scales:
            vs = [data.get((m, c, s)) for c in CONCEPTS]
            vs = [v[metric_idx] for v in vs if v is not None]
            if vs:
                rows.append((m, s, np.mean(vs)))
    return rows

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
for j, (title, ylim) in enumerate(zip(COL_TITLES, YLIMS)):
    ax = axes[j]
    ax.set_xscale("log")
    ax.set_xlabel("safety / SLD guidance scale  (log)")
    ax.set_ylabel(title); ax.set_ylim(ylim)
    ax.grid(which="both", alpha=0.25, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    sld_rows  = [(s, v) for (m, s, v) in _avg(j) if m == "sld"]
    ebsg_rows = [(s, v) for (m, s, v) in _avg(j) if m == "ebsg"]
    if sld_rows:
        sld_rows.sort()
        ax.plot([s for s, _ in sld_rows], [v for _, v in sld_rows],
                label="SLD-Max (sweep gs)", **SLD_STYLE)
    if ebsg_rows:
        ebsg_rows.sort()
        ax.plot([s for s, _ in ebsg_rows], [v for _, v in ebsg_rows],
                label="EBSG (sweep ss)", **EBSG_STYLE)
    if j == 0:
        ax.legend(loc="best", framealpha=0.95, edgecolor="0.7")

fig.suptitle("Scale robustness (avg of sexual + violence). EBSG stays stable; SLD's NotRel surges at high gs.",
             fontsize=11, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    p = OUT_DIR / f"scale_robustness_avg.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
