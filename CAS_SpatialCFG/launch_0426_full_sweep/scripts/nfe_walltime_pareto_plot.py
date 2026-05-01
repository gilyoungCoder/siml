"""NFE wall-clock vs SR Pareto plot.
For each (method, concept, NFE), parse VLM eval results.
Combine with timing CSV (per_img_sec_excl_load_mtime).
Produce:
  1) Main: 1x3 panel (SR/Full/NR), concept-averaged, x=time, y=metric.
  2) Appendix: 7x3 grid (concept x metric).
"""
import re, csv, os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def parse_result(path):
    if not path.exists() or path.stat().st_size < 50:
        return None
    txt = path.read_text(errors="ignore")
    counts = {"Safe": 0, "Partial": 0, "Full": 0, "NotPeople": 0, "NotRelevant": 0}
    for m in PAT.finditer(txt):
        counts[m.group(1)] = int(m.group(2))
    S, P, F, NP, NR = counts["Safe"], counts["Partial"], counts["Full"], counts["NotPeople"], counts["NotRelevant"]
    T = S + P + F + NP + NR
    if T == 0:
        return None
    denom = S + P + F + (NP if NP > 0 else NR)
    sr = (S + P) / denom * 100 if denom else 0.0
    full_pct = F / T * 100
    nr_pct = (NP + NR) / T * 100
    return sr, full_pct, nr_pct


def find_result_file(method, concept, step):
    cell = DATA_DIR / f"{method}_{concept}_steps{step}"
    rubric = EVAL_NAME[concept]
    for cand in (cell / f"results_qwen3_vl_{rubric}_v5.txt",
                 cell / "all" / f"results_qwen3_vl_{rubric}_v5.txt",
                 cell / "safe" / f"results_qwen3_vl_{rubric}_v5.txt"):
        if cand.exists() and cand.stat().st_size > 50:
            return cand
    return None


# Parse all cells
data = {}  # (method, concept, step) -> (SR, Full, NR)
missing = []
for m in METHODS:
    for c in CONCEPTS:
        for s in STEPS:
            f = find_result_file(m, c, s)
            if f is None:
                missing.append((m, c, s))
                continue
            r = parse_result(f)
            if r is None:
                missing.append((m, c, s))
                continue
            data[(m, c, s)] = r

print(f"Parsed {len(data)} cells, {len(missing)} missing.")
if missing:
    print("Missing (first 20):", missing[:20])

# Parse timing CSV
timing = {}  # (method, nfe) -> per_img_sec
with TIMING_CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        m = row["method"]
        if m == "safedenoiser_v2":
            m = "safedenoiser"
        if m == "sgf_v2":
            m = "sgf"
        nfe = int(row["nfe"])
        try:
            t = float(row["per_img_sec_excl_load_mtime"])
        except (ValueError, KeyError):
            t = float("nan")
        timing[(m, nfe)] = t

print(f"Timing entries: {len(timing)}")

# Build aggregated data
# avg over concepts per (method, step)
agg = {}
for m in METHODS:
    for s in STEPS:
        srs, fulls, nrs = [], [], []
        for c in CONCEPTS:
            r = data.get((m, c, s))
            if r is None:
                continue
            srs.append(r[0])
            fulls.append(r[1])
            nrs.append(r[2])
        if srs:
            agg[(m, s)] = (np.mean(srs), np.mean(fulls), np.mean(nrs), len(srs))

# Save aggregate CSV
csv_out = OUT_DIR / "nfe_walltime_5method_7concept_table.csv"
with csv_out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "concept", "nfe", "SR_pct", "Full_pct", "NR_pct"])
    for (m, c, s), (sr, fp, nr) in sorted(data.items()):
        w.writerow([m, c, s, f"{sr:.1f}", f"{fp:.1f}", f"{nr:.1f}"])
print(f"Wrote: {csv_out}")

agg_csv = OUT_DIR / "nfe_walltime_5method_concept_avg.csv"
with agg_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "nfe", "per_img_sec", "SR_avg_pct", "Full_avg_pct", "NR_avg_pct", "n_concepts"])
    for (m, s), (sr, fp, nr, n) in sorted(agg.items()):
        t = timing.get((m, s), float("nan"))
        w.writerow([m, s, f"{t:.4f}", f"{sr:.1f}", f"{fp:.1f}", f"{nr:.1f}", n])
print(f"Wrote: {agg_csv}")

# Plot 1: concept-averaged 1x3
COLORS = {"baseline": "#888888", "safree": "#1f77b4", "safedenoiser": "#2ca02c",
          "sgf": "#ff7f0e", "ebsg": "#d62728"}
LABELS = {"baseline": "Baseline (SD1.4)", "safree": "SAFREE",
          "safedenoiser": "SafeDenoiser", "sgf": "SGF", "ebsg": "EBSG (Ours)"}
MARKERS = {"baseline": "o", "safree": "s", "safedenoiser": "D", "sgf": "v", "ebsg": "*"}

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
metric_labels = ["SR (Safe+Partial) %", "Full violation %", "NotRel %"]
metric_idx = [0, 1, 2]
ylims = [(0, 100), (0, 80), (0, 50)]

for j, (mi, lbl, ylim) in enumerate(zip(metric_idx, metric_labels, ylims)):
    ax = axes[j]
    for m in METHODS:
        xs, ys = [], []
        for s in STEPS:
            v = agg.get((m, s))
            if v is None: continue
            t = timing.get((m, s), None)
            if t is None or np.isnan(t): continue
            xs.append(t)
            ys.append(v[mi])
        if not xs: continue
        ax.plot(xs, ys, "-", color=COLORS[m], linewidth=2.4,
                marker=MARKERS[m], markersize=11 if m == "ebsg" else 8,
                label=LABELS[m], alpha=0.9)
    ax.set_xlabel("per-image generation time (s)", fontsize=12)
    ax.set_ylabel(lbl, fontsize=12)
    ax.set_xscale("log")
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3, which="both")
    if j == 0:
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

fig.suptitle("Wall-clock cost vs concept-averaged metrics (I2P q16 top-60, 7 concepts × 8 NFE)\n"
             "X-axis: per-image gen time (excl. model load), measured on siml-05 g0 (RTX 3090). "
             "Markers = NFE values {5,10,15,20,25,30,40,50}.",
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = OUT_DIR / f"nfe_walltime_pareto_concept_avg.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()

# Plot 2: 7 row × 3 col facet
fig, axes = plt.subplots(7, 3, figsize=(15, 22))
for ci, c in enumerate(CONCEPTS):
    for j, (lbl, ylim) in enumerate(zip(metric_labels, ylims)):
        ax = axes[ci, j]
        for m in METHODS:
            xs, ys = [], []
            for s in STEPS:
                r = data.get((m, c, s))
                if r is None: continue
                t = timing.get((m, s), None)
                if t is None or np.isnan(t): continue
                xs.append(t)
                ys.append(r[j])
            if not xs: continue
            ax.plot(xs, ys, "-", color=COLORS[m], linewidth=2.0,
                    marker=MARKERS[m], markersize=9 if m == "ebsg" else 6,
                    label=LABELS[m], alpha=0.9)
        if ci == 0:
            ax.set_title(lbl, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{c}\n{lbl}", fontsize=10)
        if ci == len(CONCEPTS) - 1:
            ax.set_xlabel("per-image time (s)", fontsize=10)
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3, which="both")
        if ci == 0 and j == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle("Wall-clock cost vs metric — per concept (rows) × per metric (cols)",
             fontsize=12, y=1.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = OUT_DIR / f"nfe_walltime_pareto_per_concept.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()

# Print key callouts
print("\n=== Key callouts (concept-averaged) ===")
print(f"{'method':<14} {'NFE':>4} {'time(s)':>8} {'SR':>6} {'Full':>6} {'NR':>6}")
for m in METHODS:
    for s in [5, 10, 50]:
        a = agg.get((m, s))
        t = timing.get((m, s), float("nan"))
        if a is None: continue
        print(f"{m:<14} {s:>4} {t:>8.2f} {a[0]:>6.1f} {a[1]:>6.1f} {a[2]:>6.1f}")
print("Done.")
