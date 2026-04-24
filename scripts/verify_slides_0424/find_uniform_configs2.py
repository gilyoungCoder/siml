#!/usr/bin/env python3
"""Simpler search — list all anchor_inpaint cells per dataset with their config + SR."""
import json, os, glob
from collections import Counter, defaultdict

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
OUT = f"{BASE}/outputs"

def sr(jpath):
    try: d = json.load(open(jpath))
    except: return None
    cats = [v.get('category', '?') if isinstance(v, dict) else '?' for v in d.values()]
    cc = Counter(cats); n = len(cats)
    if n == 0: return None
    s = cc.get('Safe', 0); p = cc.get('Partial', 0); f = cc.get('Full', 0)
    nr = cc.get('NotRelevant', 0) + cc.get('NotPeople', 0)
    return n, s, p, f, nr, (s+p)/n*100

# Walk EVERY directory under outputs, find ones with args.json + categories_qwen3_vl_nudity_v5.json
print("Scanning all dirs with args.json + nudity v5 eval...")
records = []
for args_path in glob.glob(f"{OUT}/**/args.json", recursive=True):
    d = os.path.dirname(args_path)
    try: a = json.load(open(args_path))
    except: continue
    if a.get("how_mode") != "anchor_inpaint": continue
    prompts = a.get("prompts", "").lower()
    # Determine dataset
    if "ringabell" in prompts or "rab" in prompts.split("/")[-1].split(".")[0]:
        ds = "RAB"
    elif "unlearndiff" in prompts:
        ds = "UD"
    elif "/mma" in prompts or prompts.endswith("mma.txt"):
        ds = "MMA"
    elif "p4dn" in prompts:
        ds = "P4DN"
    else:
        continue
    # v5 json
    jpath = os.path.join(d, "categories_qwen3_vl_nudity_v5.json")
    if not os.path.exists(jpath): continue
    r = sr(jpath)
    if r is None: continue
    n, s, p, f, nr, srv = r
    expected = {"RAB":79, "UD":142, "MMA":[999,1000], "P4DN":151}[ds]
    if isinstance(expected, list):
        if n not in expected: continue
    else:
        if n != expected and abs(n - expected) > 2: continue
    cfg = {
        "ss": a.get("safety_scale"),
        "τ_t": a.get("attn_threshold"),
        "τ_i": a.get("img_attn_threshold"),
        "τ_cas": a.get("cas_threshold"),
        "probe": a.get("probe_mode"),
        "pack": os.path.basename(os.path.dirname(a.get("family_config", "?"))) if a.get("family_config") else None,
    }
    records.append((ds, srv, n, cfg, d.replace(OUT+"/", "")))

print(f"Total complete anchor_inpaint nudity v5 records: {len(records)}\n")

# Group by config
by_cfg = defaultdict(dict)
for ds, srv, n, cfg, dirn in records:
    key = (cfg["ss"], cfg["τ_t"], cfg["τ_i"], cfg["probe"], cfg["pack"])
    if ds not in by_cfg[key] or srv > by_cfg[key][ds][0]:
        by_cfg[key][ds] = (srv, n, dirn, cfg["τ_cas"])

print("="*100)
print("# CONFIGS COVERING ALL 4 DATASETS (UD + RAB + MMA + P4DN)")
print("="*100)
ranked = []
for key, dsmap in by_cfg.items():
    if {"UD","RAB","MMA","P4DN"} <= set(dsmap.keys()):
        avg = sum(dsmap[d][0] for d in ["UD","RAB","MMA","P4DN"])/4
        ranked.append((avg, key, dsmap))
ranked.sort(reverse=True)
print(f"\nFound {len(ranked)} uniform configs.\n")
for i, (avg, key, dsmap) in enumerate(ranked[:5]):
    ss, tt, ti, probe, pack = key
    print(f"## Rank {i+1}: avg={avg:.2f}% — ss={ss} τ_t={tt} τ_i={ti} probe={probe} pack={pack}")
    for ds in ["UD","RAB","MMA","P4DN"]:
        srv, n, dirn, tcas = dsmap[ds]
        print(f"   {ds:5}: SR={srv:5.1f}% n={n} τ_cas={tcas}  {dirn}")
    print()

print("\n" + "="*100)
print("# CONFIGS COVERING 3 OF 4 DATASETS")
print("="*100)
near = []
for key, dsmap in by_cfg.items():
    if len(dsmap) == 3 and {"UD","RAB","MMA","P4DN"} > set(dsmap.keys()):
        avg = sum(dsmap[d][0] for d in dsmap)/3
        near.append((avg, key, dsmap))
near.sort(reverse=True)
print(f"\nFound {len(near)} 3-of-4 configs. Top 5:\n")
for i, (avg, key, dsmap) in enumerate(near[:5]):
    ss, tt, ti, probe, pack = key
    missing = {"UD","RAB","MMA","P4DN"} - set(dsmap.keys())
    print(f"## avg(3)={avg:.2f}% — ss={ss} τ_t={tt} τ_i={ti} probe={probe} pack={pack}  MISSING={missing}")
    for ds in sorted(dsmap.keys()):
        srv, n, dirn, tcas = dsmap[ds]
        print(f"   {ds:5}: SR={srv:5.1f}% n={n}  {dirn}")
    print()

print("\n" + "="*100)
print("# PER-DATASET TOP CELLS (any config)")
print("="*100)
for ds_target in ["UD", "RAB", "MMA", "P4DN"]:
    rows = [r for r in records if r[0] == ds_target]
    rows.sort(key=lambda x: -x[1])
    print(f"\n--- {ds_target} top 5 ---")
    for ds, srv, n, cfg, dirn in rows[:5]:
        print(f"  SR={srv:5.1f}% n={n} ss={cfg['ss']} τ_t={cfg['τ_t']} τ_i={cfg['τ_i']} probe={cfg['probe']} pack={cfg['pack']}  {dirn}")
