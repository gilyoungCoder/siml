"""Polished NFE wall-clock vs SR Pareto plot for paper.
- NeurIPS-style typography (serif), grayscale-distinguishable line styles.
- EBSG (Ours) emphasized: red, bold, larger marker.
- Tighter Y limits per panel.
- 1x3 panel (SR / Full / NotRel), concept-averaged. Log-x.
- Two annotated callouts on the SR panel showing equal-time / equal-SR comparisons.
"""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep")
DATA_DIR = ROOT / "outputs/phase_nfe_walltime_v3"
TIMING_CSV = ROOT / "paper_results/figures/nfe_walltime_timing.csv"
OUT_DIR = ROOT / "paper_results/figures"

CONCEPTS = ["sexual", "violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]
STEPS = [5, 10, 15, 20, 25, 30, 40, 50]
METHODS = ["baseline", "safree", "safedenoiser", "sgf", "ebsg"]
EVAL_NAME = {"sexual": "nudity", "violence": "violence", "self-harm": "self_harm",
             "shocking": "shocking", "illegal_activity": "illegal",
             "harassment": "harassment", "hate": "hate"}

PAT = re.compile(r"-\s*(Safe|Partial|Full|NotPeople|NotRelevant):\s*(\d+)\s*\(([-0-9.]+)%\)")


def parse_result(p):
    if not p.exists() or p.stat().st_size < 50:
        return None
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
    cell = DATA_DIR / f"{method}_{c}_steps{s}"
    rub = EVAL_NAME[c]
    for cand in (cell / f"results_qwen3_vl_{rub}_v5.txt",
                 cell / "all" / f"results_qwen3_vl_{rub}_v5.txt",
                 cell / "safe" / f"results_qwen3_vl_{rub}_v5.txt"):
        if cand.exists() and cand.stat().st_size > 50:
            return cand
    return None


# Parse data
data = {}
for m in METHODS:
    for c in CONCEPTS:
        for s in STEPS:
            f = find_result(m, c, s)
            if f is None: continue
            r = parse_result(f)
            if r is None: continue
            data[(m, c, s)] = r

# Concept-averaged
agg = {}
for m in METHODS:
    for s in STEPS:
        vs = [data.get((m, c, s)) for c in CONCEPTS]
        vs = [v for v in vs if v is not None]
        if vs:
            agg[(m, s)] = (np.mean([v[0] for v in vs]),
                           np.mean([v[1] for v in vs]),
                           np.mean([v[2] for v in vs]))

# Timing
timing = {}
with TIMING_CSV.open() as f:
    for row in csv.DictReader(f):
        m = row["method"].replace("_v2", "")
        nfe = int(row["nfe"])
        try:
            timing[(m, nfe)] = float(row["per_img_sec_excl_load_mtime"])
        except (ValueError, KeyError):
            pass

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Method styles: grayscale-friendly via line style + marker shape, color emphasizes EBSG
STYLE = {
    "baseline":     dict(color="#7f7f7f", linestyle="-",  linewidth=1.6, marker="o", markersize=6,  alpha=0.85, zorder=2),
    "safree":       dict(color="#1f4e8a", linestyle="--", linewidth=1.6, marker="s", markersize=6,  alpha=0.85, zorder=2),
    "safedenoiser": dict(color="#2a8c4a", linestyle="-.", linewidth=1.6, marker="D", markersize=6,  alpha=0.85, zorder=2),
    "sgf":          dict(color="#c46a00", linestyle=":",  linewidth=1.8, marker="v", markersize=7,  alpha=0.85, zorder=2),
    "ebsg":         dict(color="#c11c1c", linestyle="-",  linewidth=2.8, marker="*", markersize=14, alpha=1.0,  zorder=5),
}
LABELS = {"baseline": "Baseline", "safree": "SAFREE", "safedenoiser": "SafeDenoiser",
          "sgf": "SAFREE+SGF", "ebsg": r"\textbf{EBSG (Ours)}"}
# strip latex if not enabled
LABELS = {"baseline": "Baseline", "safree": "SAFREE", "safedenoiser": "SafeDenoiser",
          "sgf": "SAFREE+SGF", "ebsg": "EBSG (Ours)"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
metric_titles = ["SR (Safe + Partial) %  ↑", "Full violation %  ↓", "NotRelevant %  ↓"]
ylims = [(0, 80), (0, 60), (0, 100)]

for j, (title, ylim) in enumerate(zip(metric_titles, ylims)):
    ax = axes[j]
    for m in METHODS:
        xs, ys = [], []
        for s in STEPS:
            v = agg.get((m, s))
            t = timing.get((m, s))
            if v is None or t is None: continue
            xs.append(t); ys.append(v[j])
        if not xs: continue
        ax.plot(xs, ys, label=LABELS[m], **STYLE[m])
    ax.set_xlabel("per-image generation time (s)")
    ax.set_ylabel(title)
    ax.set_xlim(0, 15)
    ax.set_ylim(ylim)
    ax.grid(which="both", alpha=0.25, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if j == 0:
        ax.legend(loc="lower right", framealpha=0.95, edgecolor="0.7")

fig.suptitle(
    "Wall-clock vs metric trade-off (concept-averaged across 7 I2P concepts; markers = NFE ∈ {5,10,15,20,25,30,40,50}).",
    fontsize=11, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    p = OUT_DIR / f"nfe_walltime_pareto_polished.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
