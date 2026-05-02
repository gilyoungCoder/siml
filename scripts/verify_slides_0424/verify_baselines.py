#!/usr/bin/env python3
"""Verify baselines/SAFREE/training-based methods for nudity main + MJA cross-backbone."""
import json, os, time, glob
from collections import Counter

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
OUT = f"{BASE}/outputs"
SAFREE_DIR = "/mnt/home3/yhgil99/unlearning/SAFREE"

def sr(jpath):
    try:
        d = json.load(open(jpath))
    except: return None
    cats = [v.get('category', '?') if isinstance(v, dict) else '?' for v in d.values()]
    cc = Counter(cats); n = len(cats)
    if n == 0: return None
    s = cc.get('Safe', 0); p = cc.get('Partial', 0); f = cc.get('Full', 0)
    nr = cc.get('NotRelevant', 0) + cc.get('NotPeople', 0)
    return n, s, p, f, nr, (s+p)/n*100

def fmt_mtime(p):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))

def report(label, jpath):
    r = sr(jpath)
    if r is None:
        print(f"  {label}: PARSE_ERROR ({jpath})")
        return
    n, s, p, f, nr, srv = r
    rel = jpath.split("outputs/")[-1] if "outputs/" in jpath else jpath
    print(f"  {label}: SR={srv:5.1f}% n={n}  [{fmt_mtime(jpath)}]  {rel}")

print("="*100)
print("NUDITY BENCHMARKS — Baseline / SAFREE / SLD / ESD / SDD / RECE")
print("="*100)
DATASETS = ["ringabell", "unlearndiff", "mma", "p4dn"]
LABEL_MAP = {"ringabell":"RAB", "unlearndiff":"UD", "mma":"MMA", "p4dn":"P4DN"}

for ds in DATASETS:
    print(f"\n--- Dataset: {LABEL_MAP[ds]} ({ds}) ---")
    # Baseline
    for cand in glob.glob(f"{OUT}/launch_0420_nudity/baseline_sd14*/{ds}/categories_qwen3_vl_nudity*v5*.json") + \
                glob.glob(f"{OUT}/launch_0420_nudity/baseline_sd14*/{ds}/categories_qwen3_vl_nudity_v3.json") + \
                glob.glob(f"{OUT}/launch_0420_nudity/baseline_sd14*/{ds}/categories_qwen3_vl_nudity.json"):
        report(f"BASELINE/{LABEL_MAP[ds]}", cand)
    # SAFREE
    for cand in glob.glob(f"{OUT}/launch_0420_nudity/safree_sd14*/{ds}/categories_qwen3_vl_nudity*v5*.json") + \
                glob.glob(f"{OUT}/launch_0420_nudity/safree_sd14*/{ds}/categories_qwen3_vl_nudity_v3.json") + \
                glob.glob(f"{OUT}/launch_0420_nudity/safree_sd14*/{ds}/categories_qwen3_vl_nudity.json"):
        report(f"SAFREE/{LABEL_MAP[ds]}", cand)
    # SLD-Max / ESD / SDD / RECE - look in different places
    for kind in ["sld_max", "sld", "esd", "sdd", "rece", "uce"]:
        for pattern in [f"{OUT}/**/{kind}*/{ds}/categories*v5*.json",
                        f"{OUT}/**/{kind}*sd14*/{ds}/categories*v5*.json",
                        f"{OUT}/**/{ds}/{kind}*/categories*v5*.json"]:
            for cand in glob.glob(pattern, recursive=True):
                report(f"{kind.upper()}/{LABEL_MAP[ds]}", cand)

print()
print("="*100)
print("MJA CROSS-BACKBONE — Baseline / SAFREE per (backbone, concept)")
print("="*100)
CONCEPTS = ["sexual", "violent", "illegal", "disturbing"]
BACKBONES = ["sd14", "sd3", "flux1"]

for back in BACKBONES:
    print(f"\n--- Backbone: {back} ---")
    for c in CONCEPTS:
        # Baseline
        for cand in glob.glob(f"{OUT}/launch_0420/baseline_{back}/mja_{c}/categories_qwen3_vl_*v5*.json") + \
                    glob.glob(f"{OUT}/launch_0420/baseline_{back}/mja_{c}*/categories_qwen3_vl_*v5*.json"):
            report(f"BASELINE/{back}/{c}", cand)
        # SAFREE
        for cand in glob.glob(f"{OUT}/launch_0420/safree_{back}/mja_{c}/categories_qwen3_vl_*v5*.json") + \
                    glob.glob(f"{OUT}/launch_0420/safree_{back}/mja_{c}*/categories_qwen3_vl_*v5*.json"):
            report(f"SAFREE/{back}/{c}", cand)

print()
print("="*100)
print("EXTERNAL BASELINES — search whole tree for SLD/ESD/SDD/RECE/UCE/SDD reproductions")
print("="*100)
search_terms = ["esd", "sdd", "uce", "rece", "sld_max", "sld_strong", "sld_medium", "sld_weak"]
for term in search_terms:
    matches = []
    # Look in CAS_SpatialCFG/outputs
    for d in glob.glob(f"{OUT}/**/{term}*", recursive=True):
        if os.path.isdir(d):
            jsons = glob.glob(f"{d}/**/categories_qwen3_vl_nudity*v5*.json", recursive=True)
            matches.extend(jsons[:3])  # cap
    # Also look in SAFREE_DIR
    for d in glob.glob(f"{SAFREE_DIR}/**/{term}*", recursive=True):
        if os.path.isdir(d):
            jsons = glob.glob(f"{d}/**/categories_qwen3_vl_nudity*v5*.json", recursive=True)
            matches.extend(jsons[:3])
    if matches:
        print(f"\n[{term}]")
        for jp in matches[:8]:
            report(f"  found", jp)
