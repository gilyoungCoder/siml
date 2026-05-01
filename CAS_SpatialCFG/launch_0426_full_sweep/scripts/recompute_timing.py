#!/usr/bin/env python3
"""Re-compute per-image generation time from filesystem mtimes.
The original timing CSVs had a bug (recorded END timestamp instead of elapsed wall).
Use PNG mtime range as ground-truth gen duration.
"""
import os, csv, glob

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_safedenoiser_sgf"
OUT_CSV = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures/nfe_sd_sgf_timing.csv"

METHODS = ["safedenoiser", "sgf"]
CONCEPTS = ["violence", "shocking", "self-harm", "sexual"]
STEPS = [1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]

rows = []
for m in METHODS:
    for c in CONCEPTS:
        for s in STEPS:
            d = f"{BASE}/{m}_{c}_step{s}/all"
            pngs = sorted(glob.glob(f"{d}/*.png"), key=os.path.getmtime)
            if len(pngs) < 2:
                rows.append((m, c, s, len(pngs), None, None, None))
                continue
            mtimes = [os.path.getmtime(p) for p in pngs]
            first = min(mtimes)
            last  = max(mtimes)
            wall  = last - first  # gen duration after first PNG (excludes initial model load)
            n     = len(pngs)
            per_img = wall / (n - 1)  # use n-1 since the first PNG is t=0 reference
            rows.append((m, c, s, n, round(wall, 2), round(per_img, 3), round(per_img/s, 4)))

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","concept","step","n_imgs","wall_sec","per_img_sec","per_step_per_img_sec"])
    for r in rows:
        w.writerow(r)
print(f"Saved {OUT_CSV}")

# Quick averages per (method, step) for overall framing
import collections
avg_per_img = collections.defaultdict(list)
for r in rows:
    if r[5] is not None:
        avg_per_img[(r[0], r[2])].append(r[5])
print()
print("Avg per-image time per (method, step):")
print("step  | safedenoiser | sgf")
print("------|--------------|--------")
for s in STEPS:
    sd = avg_per_img.get(("safedenoiser", s), [])
    sg = avg_per_img.get(("sgf", s), [])
    sd_avg = sum(sd)/len(sd) if sd else None
    sg_avg = sum(sg)/len(sg) if sg else None
    print(f"{s:>5} | {sd_avg:>12.3f} | {sg_avg:>6.3f}" if sd_avg and sg_avg else f"{s:>5} | NA           | NA")
