#!/usr/bin/env python3
"""Compute FID for COCO FID experiment.
Compares each method's generated images vs COCO val real images.
"""
import os, sys, json, glob
from pathlib import Path

REPO = "/mnt/home3/yhgil99/unlearning"
GEN_BASE = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_coco_fid"
OUT = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures"
Path(OUT).mkdir(parents=True, exist_ok=True)

# Aggregate generated images per method into single dir for FID
import shutil
METHODS = ["ebsg", "safree", "baseline"]

# Reference: COCO val real images. Try common paths.
REF_CANDIDATES = [
    f"{REPO}/datasets/coco/val2017_5k",
    f"{REPO}/datasets/coco_val",
    f"{REPO}/CAS_SpatialCFG/datasets/coco_val",
]
REF_DIR = None
for c in REF_CANDIDATES:
    if os.path.isdir(c) and len(list(Path(c).glob("*.jpg"))) > 100:
        REF_DIR = c; break
if REF_DIR is None:
    print("ERROR: no COCO val reference dir found. Looked at:", REF_CANDIDATES)
    sys.exit(1)

# For each method, collect all PNGs from slot dirs
agg_dirs = {}
for method in METHODS:
    method_dir = f"{GEN_BASE}/{method}"
    flat = f"{GEN_BASE}/_flat_{method}"
    Path(flat).mkdir(exist_ok=True)
    pngs = list(Path(method_dir).rglob("*.png"))
    print(f"  {method}: {len(pngs)} PNGs")
    if len(pngs) < 100:
        print(f"    [skip {method}: too few]")
        continue
    # Symlink to flat dir
    for p in pngs:
        link = f"{flat}/{p.parent.name}_{p.name}"
        try:
            if not os.path.exists(link):
                os.symlink(str(p), link)
        except Exception as e:
            shutil.copy(p, link)
    agg_dirs[method] = flat

# Compute FID via clean-fid or pytorch-fid
print(f"\nCOCO ref: {REF_DIR}")
print(f"Methods aggregated: {list(agg_dirs.keys())}")

# Try clean-fid first (best)
try:
    from cleanfid import fid
    USE = "clean-fid"
    fid_fn = lambda gd, rd: fid.compute_fid(gd, rd, mode="clean")
except ImportError:
    try:
        # fallback: pytorch-fid via subprocess
        import subprocess
        USE = "pytorch-fid"
        def fid_fn(gd, rd):
            r = subprocess.run(["python","-m","pytorch_fid", gd, rd],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if "FID:" in line:
                    return float(line.split("FID:")[-1].strip())
            return None
    except Exception:
        print("ERROR: neither clean-fid nor pytorch-fid available. Install: pip install clean-fid")
        sys.exit(1)

print(f"Using: {USE}\n")
results = {}
for method, gd in agg_dirs.items():
    print(f"  computing FID({method}, COCO_val)...")
    score = fid_fn(gd, REF_DIR)
    results[method] = score
    print(f"  FID({method}) = {score:.2f}")

with open(f"{OUT}/coco_fid.json","w") as f:
    json.dump({"reference": REF_DIR, "results": results}, f, indent=2)
print(f"\nsaved: {OUT}/coco_fid.json")
