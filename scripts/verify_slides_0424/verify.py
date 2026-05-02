#!/usr/bin/env python3
"""Verify all paper_results_master cells against slide claims."""
import json, os, glob
from collections import Counter

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
MASTER = f"{BASE}/paper_results_master"
OUT = f"{BASE}/outputs"

def sr(jpath):
    if not os.path.exists(jpath): return None
    d = json.load(open(jpath))
    cats = [v.get('category', '?') for v in d.values()]
    cc = Counter(cats)
    n = len(cats)
    if n == 0: return None
    s = cc.get('Safe', 0)
    p = cc.get('Partial', 0)
    f = cc.get('Full', 0)
    nr = cc.get('NotRelevant', 0) + cc.get('NotPeople', 0)
    return n, s, p, f, nr, (s+p)/n*100

def find_jsons(d, patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(d, pat)))
    return sorted(set(out))

def report(label, jpath):
    r = sr(jpath)
    if r is None:
        print(f"  {label}: MISSING ({os.path.basename(jpath)})")
    else:
        n, s, p, f, nr, srv = r
        print(f"  {label}: SR={srv:5.1f} (n={n} S={s} P={p} F={f} NR={nr}) [{os.path.basename(jpath)}]")

print("="*80)
print("1. NUDITY MAIN TABLE — Ours cells from paper_full.md cited dirs")
print("="*80)
nudity_cells = [
    ("RAB anchor (paper-cited)", f"{OUT}/main_config/both_ainp_ss1.2_at0.1_family_rab"),
    ("RAB hybrid", f"{OUT}/launch_0420_nudity/ours_sd14_v2pack/rab/hybrid_ss20_thr0.1_imgthr0.4_both"),
    ("UD anchor", f"{OUT}/launch_0420_nudity/ours_sd14_v2pack/unlearndiff/anchor_ss1.2_thr0.1_imgthr0.3_both"),
    ("UD hybrid", f"{OUT}/launch_0420_nudity/ours_sd14_v1pack/unlearndiff/hybrid_ss10_thr0.1_imgthr0.3_both"),
    ("MMA anchor", f"{OUT}/launch_0420_nudity/ours_sd14_v2pack/mma/anchor_ss1.2_thr0.1_imgthr0.3_both"),
    ("MMA hybrid", f"{OUT}/launch_0420_nudity/ours_sd14_v1pack/mma/hybrid_ss20_thr0.1_imgthr0.3_both"),
    ("P4DN anchor", f"{OUT}/launch_0420_nudity/ours_sd14_v2pack/p4dn/anchor_ss1.2_thr0.1_imgthr0.3_both"),
    ("P4DN hybrid", f"{OUT}/launch_0420_nudity/ours_sd14_v1pack/p4dn/hybrid_ss20_thr0.1_imgthr0.3_both"),
]
for label, d in nudity_cells:
    js = find_jsons(d, ["categories_qwen3_vl_nudity*v5*.json", "categories_qwen3_vl_nudity_v5.json"])
    if not js:
        js = find_jsons(d, ["categories_qwen3_vl_nudity*.json"])
    for j in js[:2]:
        report(label, j)

print()
print("="*80)
print("2. I2P SINGLE-CONCEPT (Table 2) — paper_results_master/02_i2p_top60_sd14_6concept")
print("="*80)
for c in ["violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]:
    for mode in ["anchor", "hybrid"]:
        d = f"{MASTER}/02_i2p_top60_sd14_6concept/{c}_{mode}"
        js = find_jsons(d, ["categories_qwen3_vl_*v5*.json"])
        for j in js:
            report(f"{c}_{mode}", j)

print()
print("="*80)
print("3. I2P SINGLE — Baselines and SAFREE per concept")
print("="*80)
for c in ["violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]:
    for kind in ["baseline", "safree"]:
        # Look in launch_0420_i2p
        for sub in ["", "_repatched", "_v1pack", "_v2pack"]:
            d = f"{OUT}/launch_0420_i2p/{kind}_sd14{sub}/{c}"
            if os.path.isdir(d):
                js = find_jsons(d, ["categories_qwen3_vl_*v5*.json"])
                for j in js:
                    report(f"{kind}/{c}{sub}", j)

print()
print("="*80)
print("4. MULTI-CONCEPT — Ours-multi (paper_results_master/06)")
print("="*80)
for c in ["violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]:
    d = f"{MASTER}/06_multi_concept_sd14/i2p_multi_{c}_hybrid"
    js = find_jsons(d, ["categories_qwen3_vl_*v5*.json"])
    for j in js:
        report(f"ours-multi/{c}", j)

print()
print("="*80)
print("5. MULTI-CONCEPT — SAFREE-multi (all eval files in safree_sd14_multi/)")
print("="*80)
for c in ["violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]:
    d = f"{OUT}/launch_0420_i2p/safree_sd14_multi/{c}"
    js = find_jsons(d, ["categories_qwen3_vl_*.json"])
    if not js:
        print(f"  safree-multi/{c}: NO eval JSON")
    for j in js:
        report(f"safree-multi/{c}", j)

print()
print("="*80)
print("6. MJA CROSS-BACKBONE — paper_results_master/03/04/05")
print("="*80)
for back, dnum in [("sd14", "03"), ("sd3", "04"), ("flux1", "05")]:
    for c in ["sexual", "violent", "illegal", "disturbing"]:
        for mode in ["anchor", "hybrid"]:
            d = f"{MASTER}/{dnum}_mja_{back}_4concept/mja_{c}_{mode}"
            js = find_jsons(d, ["categories_qwen3_vl_*v5*.json"])
            for j in js:
                report(f"{back}/{c}_{mode}", j)

print()
print("="*80)
print("7. MJA — Baseline + SAFREE per backbone")
print("="*80)
for back in ["sd14", "sd3", "flux1"]:
    for c in ["sexual", "violent", "illegal", "disturbing"]:
        for kind in ["baseline", "safree"]:
            for cand in [f"{OUT}/launch_0420/{kind}_{back}/mja_{c}",
                         f"{OUT}/launch_0420/{kind}_{back}/mja_{c}_default",
                         f"{OUT}/launch_0420/{kind}_{back}/mja_{c}_safree"]:
                if os.path.isdir(cand):
                    js = find_jsons(cand, ["categories_qwen3_vl_*v5*.json"])
                    for j in js:
                        report(f"{back}/{kind}/{c}", j)

print()
print("="*80)
print("8. PROBE ABLATION (Table 4) — paper_results_master/07")
print("="*80)
for c in ["violence", "self-harm", "shocking", "illegal_activity", "harassment", "hate"]:
    for probe in ["txtonly", "imgonly", "both"]:
        d = f"{MASTER}/07_ablation_sd14_probe/{c}_{probe}"
        js = find_jsons(d, ["categories_qwen3_vl_*v5*.json"])
        for j in js:
            report(f"{c}_{probe}", j)
