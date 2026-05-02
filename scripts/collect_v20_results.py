#!/usr/bin/env python3
"""Collect v20 Qwen3-VL results."""
import json, os

base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v20"
results = []
for d in sorted(os.listdir(base)):
    rf = os.path.join(base, d, "categories_qwen3_vl_nudity.json")
    if not os.path.exists(rf):
        continue
    data = json.load(open(rf))
    total = len(data)
    cats = {}
    for v in data.values():
        c = v.get("category", "Unknown")
        cats[c] = cats.get(c, 0) + 1
    safe = cats.get("Safe", 0)
    partial = cats.get("Partial", 0)
    nr = cats.get("NotRel", 0)
    full = cats.get("Full", 0)
    sr = 100 * (safe + partial) / total if total else 0
    nrp = 100 * nr / total if total else 0
    fp = 100 * full / total if total else 0
    results.append((d, total, sr, nrp, fp))

print("=" * 65)
print("v20 Qwen3-VL Results (Ring-A-Bell, 1 sample/prompt)")
print("=" * 65)
header = f"{'Config':35s} {'SR%':>6s} {'NR%':>6s} {'Full%':>6s}"
print(header)
print("-" * 65)
for d, total, sr, nrp, fp in sorted(results, key=lambda x: -x[2]):
    print(f"{d:35s} {sr:5.1f}% {nrp:5.1f}% {fp:5.1f}%")
