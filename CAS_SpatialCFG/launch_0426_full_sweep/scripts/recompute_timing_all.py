#!/usr/bin/env python3
"""Re-compute per-image generation time for ALL 5 methods (EBSG/SAFREE/baseline + SafeDenoiser/SGF)."""
import os, csv, glob, collections

BASE_OURS  = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_full"
BASE_SDSGF = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_safedenoiser_sgf"
OUT_CSV    = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures/nfe_5method_timing.csv"

CONCEPTS = ["violence", "shocking", "self-harm", "sexual"]
STEPS    = [1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]

def measure(d):
    pngs = sorted(glob.glob(f"{d}/*.png"), key=os.path.getmtime)
    if len(pngs) < 2: return len(pngs), None, None
    mtimes = [os.path.getmtime(p) for p in pngs]
    wall = max(mtimes) - min(mtimes)
    n = len(pngs)
    per_img = wall / (n - 1)
    return n, round(wall, 2), round(per_img, 3)

rows = []
# EBSG / SAFREE / baseline → phase_nfe_full/{method}_{concept}_steps{N}/
for m in ["ebsg", "safree", "baseline"]:
    for c in CONCEPTS:
        for s in STEPS:
            d = f"{BASE_OURS}/{m}_{c}_steps{s}"
            n, w, pi = measure(d)
            rows.append((m, c, s, n, w, pi))

# SafeDenoiser / SGF → phase_nfe_safedenoiser_sgf/{method}_{concept}_step{N}/all/
for m in ["safedenoiser", "sgf"]:
    for c in CONCEPTS:
        for s in STEPS:
            d = f"{BASE_SDSGF}/{m}_{c}_step{s}/all"
            n, w, pi = measure(d)
            rows.append((m, c, s, n, w, pi))

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","concept","step","n_imgs","wall_sec","per_img_sec"])
    for r in rows: w.writerow(r)
print(f"Saved {OUT_CSV}")

# Average per (method, step) across concepts
avg = collections.defaultdict(list)
for r in rows:
    if r[5] is not None:
        avg[(r[0], r[2])].append(r[5])
print()
print("Avg per-image gen time (sec) per (method, step) — averaged across 4 concepts:")
print(f"{'step':>5} | {'EBSG':>8} | {'SAFREE':>8} | {'baseline':>8} | {'SafeDenoiser':>12} | {'SGF':>8}")
print("-"*70)
for s in STEPS:
    parts = [f"{s:>5}"]
    for m in ["ebsg","safree","baseline","safedenoiser","sgf"]:
        v = avg.get((m, s), [])
        avg_v = sum(v)/len(v) if v else None
        parts.append(f"{avg_v:>8.3f}" if avg_v is not None else f"{'NA':>8}")
    print(" | ".join(parts))
