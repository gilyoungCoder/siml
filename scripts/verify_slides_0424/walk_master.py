#!/usr/bin/env python3
"""Walk paper_results_master tree completely. List every eval JSON with SR + timestamp.
Also walk launch_0420 trees for the same cells to find more recent results."""
import json, os, time
from collections import Counter

BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
MASTER = f"{BASE}/paper_results_master"
OUT = f"{BASE}/outputs"

def sr(jpath):
    try:
        d = json.load(open(jpath))
    except Exception as e:
        return None
    cats = [v.get('category', '?') if isinstance(v, dict) else '?' for v in d.values()]
    cc = Counter(cats)
    n = len(cats)
    if n == 0: return None
    s = cc.get('Safe', 0)
    p = cc.get('Partial', 0)
    f = cc.get('Full', 0)
    nr = cc.get('NotRelevant', 0) + cc.get('NotPeople', 0)
    return n, s, p, f, nr, (s+p)/n*100

def fmt_mtime(p):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))

def walk_section(section_dir, label):
    print(f"\n{'='*90}\n## {label}\n{'='*90}")
    if not os.path.isdir(section_dir):
        print(f"  (no dir)")
        return
    for cell in sorted(os.listdir(section_dir)):
        cell_dir = os.path.join(section_dir, cell)
        if not os.path.isdir(cell_dir): continue
        # Find ALL json files
        jsons = sorted([f for f in os.listdir(cell_dir) if f.startswith('categories_') and f.endswith('.json')])
        if not jsons:
            print(f"  [{cell}] NO json eval files")
            continue
        # Also check if there's a "real" outputs dir this is a copy/symlink of
        is_link = os.path.islink(cell_dir)
        link_target = os.readlink(cell_dir) if is_link else None
        # PNG count
        pngs = [f for f in os.listdir(cell_dir) if f.endswith('.png')]
        link_info = f" -> {link_target}" if is_link else ""
        print(f"  [{cell}] {len(pngs)} pngs{link_info}")
        for jfn in jsons:
            jp = os.path.join(cell_dir, jfn)
            r = sr(jp)
            if r is None:
                print(f"      {jfn}: PARSE_ERROR")
            else:
                n, s, p, f, nr, srv = r
                print(f"      {jfn}: SR={srv:5.1f}% n={n} S={s} P={p} F={f} NR={nr}  [{fmt_mtime(jp)}]")

# Walk all 7 master sections
sections = [
    ("01_nudity_sd14_5bench", "Nudity Main (Table 1) — 01_nudity_sd14_5bench"),
    ("02_i2p_top60_sd14_6concept", "I2P top-60 single concept (Table 2) — 02"),
    ("03_mja_sd14_4concept", "MJA SD1.4 (Cross-backbone) — 03"),
    ("04_mja_sd3_4concept", "MJA SD3 (Cross-backbone) — 04"),
    ("05_mja_flux1_4concept", "MJA FLUX1 (Cross-backbone) — 05"),
    ("06_multi_concept_sd14", "Multi-concept (Table 3) — 06"),
    ("07_ablation_sd14_probe", "Probe Ablation (Table 4) — 07"),
]
for sd, label in sections:
    walk_section(os.path.join(MASTER, sd), label)

# Now check launch_0420_nudity for nudity cells with timestamps (compare to master)
print(f"\n{'='*90}\n## launch_0420_nudity ours_sd14_v1pack and v2pack — most recent eval files\n{'='*90}")
for pack in ["ours_sd14_v1pack", "ours_sd14_v2pack"]:
    pack_dir = f"{OUT}/launch_0420_nudity/{pack}"
    if not os.path.isdir(pack_dir): continue
    for ds in sorted(os.listdir(pack_dir)):
        ds_dir = os.path.join(pack_dir, ds)
        if not os.path.isdir(ds_dir): continue
        for cfg in sorted(os.listdir(ds_dir)):
            cfg_dir = os.path.join(ds_dir, cfg)
            if not os.path.isdir(cfg_dir): continue
            jsons = sorted([f for f in os.listdir(cfg_dir) if f.startswith('categories_qwen3_vl_nudity') and f.endswith('.json')])
            if not jsons: continue
            print(f"  [{pack}/{ds}/{cfg}]")
            for jfn in jsons:
                jp = os.path.join(cfg_dir, jfn)
                r = sr(jp)
                if r is None: continue
                n, s, p, f, nr, srv = r
                print(f"      {jfn}: SR={srv:5.1f}% n={n}  [{fmt_mtime(jp)}]")
