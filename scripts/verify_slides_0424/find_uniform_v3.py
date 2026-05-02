#!/usr/bin/env python3
"""Find uniform anchor_inpaint configs — using FULL pack path as part of config key."""
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

records = []
for args_path in glob.glob(f"{OUT}/**/args.json", recursive=True):
    d = os.path.dirname(args_path)
    try: a = json.load(open(args_path))
    except: continue
    if a.get("how_mode") != "anchor_inpaint": continue
    prompts = a.get("prompts", "").lower()
    if "ringabell" in prompts: ds = "RAB"
    elif "unlearndiff" in prompts: ds = "UD"
    elif prompts.endswith("/mma.txt"): ds = "MMA"
    elif "p4dn" in prompts: ds = "P4DN"
    else: continue
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
    pack_full = a.get("family_config", "?")
    # Use full pack path (e.g., "exemplars/i2p_v1/sexual" vs "exemplars/concepts_v2/sexual")
    pack_key = "/".join(pack_full.split("/")[-3:-1]) if pack_full != "?" else "?"
    cfg = (
        a.get("safety_scale"),
        a.get("attn_threshold"),
        a.get("img_attn_threshold"),
        a.get("probe_mode"),
        pack_key,
        tuple(a.get("target_concepts", [])),
        tuple(a.get("anchor_concepts", [])),
    )
    records.append((ds, srv, n, cfg, d.replace(OUT+"/", ""), a.get("cas_threshold")))

print(f"Total records: {len(records)}\n")

by_cfg = defaultdict(dict)
for ds, srv, n, cfg, dirn, tcas in records:
    if ds not in by_cfg[cfg] or srv > by_cfg[cfg][ds][0]:
        by_cfg[cfg][ds] = (srv, n, dirn, tcas)

print("="*100)
print("# CONFIGS COVERING ALL 4 DATASETS — TRULY UNIFORM (same pack + same target/anchor concepts)")
print("="*100)
ranked = []
for key, dsmap in by_cfg.items():
    if {"UD","RAB","MMA","P4DN"} <= set(dsmap.keys()):
        avg = sum(dsmap[d][0] for d in ["UD","RAB","MMA","P4DN"])/4
        ranked.append((avg, key, dsmap))
ranked.sort(reverse=True)
print(f"Found {len(ranked)} truly-uniform configs.\n")
for i, (avg, key, dsmap) in enumerate(ranked[:5]):
    ss, tt, ti, probe, pack, tc, ac = key
    print(f"## Rank {i+1}: avg={avg:.2f}% — ss={ss} τ_t={tt} τ_i={ti} probe={probe}")
    print(f"   pack={pack}  target={list(tc)}  anchor={list(ac)}")
    for ds in ["UD","RAB","MMA","P4DN"]:
        srv, n, dirn, tcas = dsmap[ds]
        print(f"   {ds:5}: SR={srv:5.1f}% n={n} τ_cas={tcas}  {dirn}")
    print()

# Configs covering 3 datasets
print("\n" + "="*100)
print("# CONFIGS COVERING 3 OF 4 DATASETS (one missing)")
print("="*100)
near = []
for key, dsmap in by_cfg.items():
    if len(dsmap) == 3 and {"UD","RAB","MMA","P4DN"} > set(dsmap.keys()):
        avg = sum(dsmap[d][0] for d in dsmap)/3
        near.append((avg, key, dsmap))
near.sort(reverse=True)
print(f"Top 5 of {len(near)}:\n")
for i, (avg, key, dsmap) in enumerate(near[:5]):
    ss, tt, ti, probe, pack, tc, ac = key
    missing = {"UD","RAB","MMA","P4DN"} - set(dsmap.keys())
    print(f"## avg(3)={avg:.2f}% — ss={ss} τ_t={tt} τ_i={ti} probe={probe}  MISSING={missing}")
    print(f"   pack={pack}  target={list(tc)}  anchor={list(ac)}")
    for ds in sorted(dsmap.keys()):
        srv, n, dirn, tcas = dsmap[ds]
        print(f"   {ds:5}: SR={srv:5.1f}% n={n}  {dirn}")
    print()

# i2p_v1/sexual — what datasets does it cover?
print("\n" + "="*100)
print("# i2p_v1/sexual pack — all anchor_inpaint cells (any dataset)")
print("="*100)
for ds, srv, n, cfg, dirn, tcas in sorted([r for r in records if "i2p_v1/sexual" in r[3][4]], key=lambda x: -x[1]):
    ss, tt, ti, probe, pack, tc, ac = cfg
    print(f"  {ds:5} SR={srv:5.1f}% n={n} ss={ss} τ_t={tt} τ_i={ti} probe={probe}  {dirn}")

print("\n" + "="*100)
print("# concepts_v2/sexual pack — all anchor_inpaint cells (any dataset)")
print("="*100)
for ds, srv, n, cfg, dirn, tcas in sorted([r for r in records if "concepts_v2/sexual" in r[3][4]], key=lambda x: -x[1]):
    ss, tt, ti, probe, pack, tc, ac = cfg
    print(f"  {ds:5} SR={srv:5.1f}% n={n} ss={ss} τ_t={tt} τ_i={ti} probe={probe}  {dirn}")
