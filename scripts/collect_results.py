#!/usr/bin/env python3
"""Collect all v2 experiment results into a summary table."""
import os, re

base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v2_experiments"
bl = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2"

def parse_qwen(path):
    if not os.path.exists(path): return None, None
    sr, full = None, None
    for line in open(path):
        if "SR" in line and "%" in line:
            m = re.search(r"([\d.]+)%", line)
            if m: sr = float(m.group(1))
        if "Full" in line and "Rate" in line:
            m = re.search(r"([\d.]+)%", line)
            if m: full = float(m.group(1))
    return sr, full

def parse_q16(path):
    if not os.path.exists(path): return None
    for line in open(path):
        if "Inappropriate:" in line:
            m = re.search(r"([\d.]+)%", line)
            if m: return float(m.group(1))
    return None

def count_imgs(d):
    return len([f for f in os.listdir(d) if f.endswith(".png")])

def find_qwen(d):
    for f in sorted(os.listdir(d)):
        if f.startswith("results_qwen"):
            sr, full = parse_qwen(os.path.join(d, f))
            if sr is not None:
                return sr, full
    return None, None

# Baselines
print("=" * 90)
print("BASELINES")
print("=" * 90)
print(f"  {'Name':25s} {'N':>6s} {'SR%':>8s} {'Full%':>8s} {'Q16%':>8s}")
print("-" * 60)
if os.path.isdir(bl):
    for name in sorted(os.listdir(bl)):
        d = os.path.join(bl, name)
        if not os.path.isdir(d): continue
        n = count_imgs(d)
        sr, full = find_qwen(d)
        q16 = parse_q16(os.path.join(d, "results_q16.txt"))
        sr_s = f"{sr:.1f}" if sr else "-"
        full_s = f"{full:.1f}" if full else "-"
        q16_s = f"{q16:.1f}" if q16 else "-"
        print(f"  {name:25s} {n:6d} {sr_s:>8s} {full_s:>8s} {q16_s:>8s}")

# Ours
print()
print("=" * 90)
print("OURS (v2 experiments)")
print("=" * 90)
if os.path.isdir(base):
    for concept in sorted(os.listdir(base)):
        cd = os.path.join(base, concept)
        if not os.path.isdir(cd): continue
        print(f"\n--- {concept} ---")
        print(f"  {'Config':55s} {'N':>6s} {'SR%':>8s} {'Full%':>8s} {'Q16%':>8s}")
        print("  " + "-" * 82)
        for exp in sorted(os.listdir(cd)):
            ed = os.path.join(cd, exp)
            if not os.path.isdir(ed): continue
            n = count_imgs(ed)
            sr, full = find_qwen(ed)
            q16 = parse_q16(os.path.join(ed, "results_q16.txt"))
            sr_s = f"{sr:.1f}" if sr else "-"
            full_s = f"{full:.1f}" if full else "-"
            q16_s = f"{q16:.1f}" if q16 else "-"
            print(f"  {exp:55s} {n:6d} {sr_s:>8s} {full_s:>8s} {q16_s:>8s}")
