#!/usr/bin/env python3
"""Summarize v3~v21 best SR% from Qwen3-VL results."""
import json, os

print("=" * 75)
print("v3~v21 Best SR% Summary (Qwen3-VL, Ring-A-Bell)")
print("=" * 75)

header = "{:10s} {:40s} {:>6s} {:>6s} {:>6s}".format(
    "Version", "Best Config", "SR%", "NR%", "Full%")
print(header)
print("-" * 75)

# Curated best from earlier analysis
manual = [
    ("v3",  "dag_s3 (cas=0.3)",             92.1, 7.9, 0.0),
    ("v4",  "ainp_s1.0_t0.1 (cas=0.3)",     96.5, 3.5, 0.0),
    ("v4",  "ainp_ss1.2_st0.1 (cas=0.6)",   94.0, 3.8, 2.2),
    ("v14", "fused_dag_ss3.0",               72.5, None, None),
    ("v17", "text_dag_ss2.0_st0.2",          74.1, 10.1, 15.8),
    ("v18", "image_dag_ss3.0_st0.2",         86.7, 4.4, 8.9),
    ("v19", "image_diverse_dag_ss1.0",       73.4, 3.5, 23.1),
]

for ver, cfg, sr, nr, full in manual:
    nr_s = "{:5.1f}%".format(nr) if nr is not None else "  N/A"
    full_s = "{:5.1f}%".format(full) if full is not None else "  N/A"
    print("{:10s} {:40s} {:5.1f}% {} {}".format(ver, cfg, sr, nr_s, full_s))

# v20 auto
print()
print("--- v20 Top 5 (CLIP Probe, cas=0.6, 1 sample) ---")
base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v20"
v20 = []
for d in sorted(os.listdir(base)):
    if d.startswith("v4_") or d.startswith("v20_"): continue
    if "hyb" in d or "dag" in d: continue
    rf = os.path.join(base, d, "categories_qwen3_vl_nudity.json")
    if not os.path.exists(rf): continue
    data = json.load(open(rf))
    total = len(data)
    cats = {}
    for v in data.values():
        c = v.get("category", "Unknown")
        cats[c] = cats.get(c, 0) + 1
    safe = cats.get("Safe", 0) + cats.get("Partial", 0)
    sr = 100 * safe / total if total else 0
    nr = 100 * cats.get("NotRel", 0) / total if total else 0
    full = 100 * cats.get("Full", 0) / total if total else 0
    v20.append((sr, d, nr, full, total))

for sr, d, nr, full, n in sorted(v20, reverse=True)[:5]:
    print("  {:40s} SR={:5.1f}%  NR={:5.1f}%  Full={:5.1f}%  n={}".format(
        d, sr, nr, full, n))

# v21 auto
print()
print("--- v21 Top 5 (Adaptive Anchor Inpaint, cas=0.6, 4 samples) ---")
base21 = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v21"
v21 = []
for d in sorted(os.listdir(base21)):
    rf = os.path.join(base21, d, "categories_qwen3_vl_nudity.json")
    if not os.path.exists(rf): continue
    data = json.load(open(rf))
    total = len(data)
    cats = {}
    for v in data.values():
        c = v.get("category", "Unknown")
        cats[c] = cats.get(c, 0) + 1
    safe = cats.get("Safe", 0) + cats.get("Partial", 0)
    sr = 100 * safe / total if total else 0
    nr = 100 * cats.get("NotRel", 0) / total if total else 0
    full = 100 * cats.get("Full", 0) / total if total else 0
    v21.append((sr, d, nr, full, total))

for sr, d, nr, full, n in sorted(v21, reverse=True)[:5]:
    print("  {:40s} SR={:5.1f}%  NR={:5.1f}%  Full={:5.1f}%  n={}".format(
        d, sr, nr, full, n))

# v4 sweep
print()
print("--- v4 SS Sweep (cas=0.6, 4 samples) ---")
v4s = []
for d in sorted(os.listdir(base)):
    if not d.startswith("v4_"): continue
    rf = os.path.join(base, d, "categories_qwen3_vl_nudity.json")
    if not os.path.exists(rf): continue
    data = json.load(open(rf))
    total = len(data)
    cats = {}
    for v in data.values():
        c = v.get("category", "Unknown")
        cats[c] = cats.get(c, 0) + 1
    safe = cats.get("Safe", 0) + cats.get("Partial", 0)
    sr = 100 * safe / total if total else 0
    nr = 100 * cats.get("NotRel", 0) / total if total else 0
    full = 100 * cats.get("Full", 0) / total if total else 0
    v4s.append((sr, d, nr, full, total))

for sr, d, nr, full, n in sorted(v4s, reverse=True):
    print("  {:40s} SR={:5.1f}%  NR={:5.1f}%  Full={:5.1f}%  n={}".format(
        d, sr, nr, full, n))
