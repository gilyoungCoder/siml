#!/usr/bin/env python3
"""Final consistent v5 verification — baseline / SAFREE / Ours, seed=00 only for MJA baselines."""
import json, os, time, glob
from collections import Counter

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
OUT = f"{BASE}/outputs"
MASTER = f"{BASE}/paper_results_master"

def sr_filtered(jpath, name_filter=None):
    """Compute SR. Optionally filter to image names matching predicate."""
    try:
        d = json.load(open(jpath))
    except: return None
    if name_filter:
        d = {k: v for k, v in d.items() if name_filter(k)}
    cats = [v.get('category', '?') if isinstance(v, dict) else '?' for v in d.values()]
    cc = Counter(cats); n = len(cats)
    if n == 0: return None
    s = cc.get('Safe', 0); p = cc.get('Partial', 0); f = cc.get('Full', 0)
    nr = cc.get('NotRelevant', 0) + cc.get('NotPeople', 0)
    return n, s, p, f, nr, (s+p)/n*100

def fmt(r, label):
    if r is None: return f"  {label}: NO DATA"
    n, s, p, f, nr, srv = r
    return f"  {label}: SR={srv:5.1f}% n={n} S={s} P={p} F={f} NR={nr}"

# Filter for seed=00 only (filename pattern: ..._00.png or _00_*.png)
def seed00(name):
    # name like "0000_00_xxx.png" — take only _00 seed
    parts = name.replace('.png', '').split('_')
    if len(parts) >= 2:
        return parts[1] == '00'
    return True

print("="*100)
print("# FINAL GROUND TRUTH — consistent v5, seed=00 only for MJA baselines (n=100 each)")
print("="*100)

# === NUDITY MAIN TABLE ===
print("\n## A. NUDITY MAIN — UD / RAB / MMA / P4DN")
NUDITY_DS = [("unlearndiff", "UD"), ("rab", "RAB"), ("mma", "MMA"), ("p4dn", "P4DN")]

print("\n### Baseline (v5)")
for ds, label in NUDITY_DS:
    j = f"{OUT}/launch_0420_nudity/baseline_sd14/{ds}/categories_qwen3_vl_nudity_v5.json"
    if os.path.exists(j):
        print(fmt(sr_filtered(j), f"BASELINE/{label}"))
    else:
        print(f"  BASELINE/{label}: MISSING ({j})")

print("\n### SAFREE (v5)")
for ds, label in NUDITY_DS:
    j = f"{OUT}/launch_0420_nudity/safree_sd14/{ds}/categories_qwen3_vl_nudity_v5.json"
    if os.path.exists(j):
        print(fmt(sr_filtered(j), f"SAFREE/{label}"))
    else:
        print(f"  SAFREE/{label}: MISSING ({j})")

print("\n### Ours (v5) — best master cell per (dataset, mode)")
NUD_OURS = [
    ("UD anchor", f"{MASTER}/01_nudity_sd14_5bench/unlearndiff_anchor/categories_qwen3_vl_nudity_v5.json"),
    ("UD hybrid", f"{MASTER}/01_nudity_sd14_5bench/unlearndiff_hybrid/categories_qwen3_vl_nudity_v5.json"),
    ("RAB anchor (main_config best)", f"{OUT}/main_config/both_ainp_ss1.2_at0.1_family_rab/categories_qwen3_vl_nudity_v5.json"),
    ("RAB hybrid", f"{MASTER}/01_nudity_sd14_5bench/rab_hybrid/categories_qwen3_vl_nudity_v5.json"),
    ("MMA anchor", f"{MASTER}/01_nudity_sd14_5bench/mma_anchor/categories_qwen3_vl_nudity_v5.json"),
    ("MMA hybrid", f"{MASTER}/01_nudity_sd14_5bench/mma_hybrid/categories_qwen3_vl_nudity_v5.json"),
    ("P4DN anchor", f"{MASTER}/01_nudity_sd14_5bench/p4dn_anchor/categories_qwen3_vl_nudity_v5.json"),
    ("P4DN hybrid", f"{MASTER}/01_nudity_sd14_5bench/p4dn_hybrid/categories_qwen3_vl_nudity_v5.json"),
]
for label, j in NUD_OURS:
    if os.path.exists(j):
        print(fmt(sr_filtered(j), label))
    else:
        print(f"  {label}: MISSING")

# === MJA CROSS-BACKBONE ===
print("\n\n## B. MJA CROSS-BACKBONE — sexual / violent / illegal / disturbing × SD14/SD3/FLUX1")
MJA_CONCEPTS = ["sexual", "violent", "illegal", "disturbing"]
JSON_NAMES = {"sexual": "nudity", "violent": "violence", "illegal": "illegal", "disturbing": "disturbing"}

for back in ["sd14", "sd3", "flux1"]:
    print(f"\n### Backbone: {back}")
    for c in MJA_CONCEPTS:
        jname = JSON_NAMES[c]
        # Baseline — filter seed=00 if n>100
        bj = f"{OUT}/launch_0420/baseline_{back}/mja_{c}/categories_qwen3_vl_{jname}_v5.json"
        if os.path.exists(bj):
            r_full = sr_filtered(bj)
            r_seed00 = sr_filtered(bj, name_filter=seed00)
            if r_full[0] > 100:
                print(fmt(r_seed00, f"BASELINE/{back}/{c} (seed00)"))
                print(fmt(r_full, f"BASELINE/{back}/{c} (all seeds)"))
            else:
                print(fmt(r_full, f"BASELINE/{back}/{c}"))
        else:
            print(f"  BASELINE/{back}/{c}: MISSING")
        # SAFREE
        sj = f"{OUT}/launch_0420/safree_{back}/mja_{c}/categories_qwen3_vl_{jname}_v5.json"
        if os.path.exists(sj):
            print(fmt(sr_filtered(sj), f"SAFREE/{back}/{c}"))
        else:
            print(f"  SAFREE/{back}/{c}: MISSING")
        # Ours from master
        for mode in ["anchor", "hybrid"]:
            section = {"sd14": "03", "sd3": "04", "flux1": "05"}[back]
            j = f"{MASTER}/{section}_mja_{back}_4concept/mja_{c}_{mode}/categories_qwen3_vl_{jname}_v5.json"
            if os.path.exists(j):
                print(fmt(sr_filtered(j), f"OURS/{back}/{c}/{mode}"))
            else:
                print(f"  OURS/{back}/{c}/{mode}: MISSING ({j})")

# === I2P TOP60 ===
print("\n\n## C. I2P TOP60 (SD1.4) — single concept (Table 2)")
I2P = ["violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]
JSON_I2P = {"violence":"violence", "self-harm":"self_harm", "shocking":"shocking",
            "illegal_activity":"illegal", "harassment":"harassment", "hate":"hate"}

for c in I2P:
    print(f"\n### Concept: {c}")
    jn = JSON_I2P[c]
    # Baseline
    bj = f"{OUT}/launch_0420_i2p/baseline_sd14/{c}/categories_qwen3_vl_{jn}_v5.json"
    if os.path.exists(bj):
        print(fmt(sr_filtered(bj), f"BASELINE/{c}"))
    # SAFREE
    sj = f"{OUT}/launch_0420_i2p/safree_sd14/{c}/categories_qwen3_vl_{jn}_v5.json"
    if os.path.exists(sj):
        print(fmt(sr_filtered(sj), f"SAFREE/{c}"))
    # Ours anchor / hybrid (master)
    for mode in ["anchor", "hybrid"]:
        oj = f"{MASTER}/02_i2p_top60_sd14_6concept/{c}_{mode}/categories_qwen3_vl_{jn}_v5.json"
        if os.path.exists(oj):
            print(fmt(sr_filtered(oj), f"OURS/{c}/{mode}"))

# === MULTI-CONCEPT ===
print("\n\n## D. MULTI-CONCEPT (SD1.4) — Table 3")
for c in I2P:
    jn = JSON_I2P[c]
    # Ours-multi
    om = f"{MASTER}/06_multi_concept_sd14/i2p_multi_{c}_hybrid/categories_qwen3_vl_{jn}_v5.json"
    if os.path.exists(om):
        print(fmt(sr_filtered(om), f"OURS-multi/{c}"))
    # SAFREE-multi
    sm = f"{OUT}/launch_0420_i2p/safree_sd14_multi/{c}/categories_qwen3_vl_{jn}_v5.json"
    if os.path.exists(sm):
        print(fmt(sr_filtered(sm), f"SAFREE-multi/{c}"))

# === PROBE ABLATION (strict, mode=hybrid) ===
print("\n\n## E. PROBE ABLATION (strict, mode=hybrid) — Table 4")
print("Note: 'both' = paper_results_master/02/{concept}_hybrid (mode-fixed=hybrid + both probe)")
for c in I2P:
    jn = JSON_I2P[c]
    print(f"\n### Concept: {c}")
    txt = f"{MASTER}/07_ablation_sd14_probe/{c}_txtonly/categories_qwen3_vl_{jn}_v5.json"
    img = f"{MASTER}/07_ablation_sd14_probe/{c}_imgonly/categories_qwen3_vl_{jn}_v5.json"
    both = f"{MASTER}/02_i2p_top60_sd14_6concept/{c}_hybrid/categories_qwen3_vl_{jn}_v5.json"
    if os.path.exists(txt): print(fmt(sr_filtered(txt), f"  txt-only"))
    if os.path.exists(img): print(fmt(sr_filtered(img), f"  img-only"))
    if os.path.exists(both): print(fmt(sr_filtered(both), f"  both (hybrid mode)"))

print("\n\n## F. EXTERNAL BASELINES (SLD/ESD/SDD/RECE/UCE) — search for v5 evals")
import subprocess
for term in ["esd", "sdd", "uce", "rece", "sld_max", "sld_strong", "sld_medium", "sld_weak"]:
    found = []
    for d in glob.glob(f"{OUT}/**/*{term}*", recursive=True):
        if os.path.isdir(d):
            for j in glob.glob(f"{d}/**/categories_qwen3_vl_*_v5.json", recursive=True):
                found.append(j)
    if found:
        print(f"\n### {term}: {len(found)} v5 JSONs found")
        for j in found[:5]: print(f"  {j}")
    else:
        print(f"\n### {term}: NO v5 evaluation found")
