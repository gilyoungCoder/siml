#!/usr/bin/env python3
"""
Find:
1. ALL nudity Ours-anchor cells (UD/RAB/MMA/P4DN) — sorted by v5 SR
2. ALL I2P self-harm_anchor / illegal_activity_hybrid / harassment_hybrid cells WITH concept-correct target_concepts
   — print best v5 SR + full args.json config
"""
import json, os, glob
from collections import Counter

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

NUDITY_BAD = {"nudity", "nude person", "naked body"}

def is_nudity_bug(target_concepts):
    if not target_concepts: return True
    return set(target_concepts) <= NUDITY_BAD or set(target_concepts) == NUDITY_BAD

print("="*100)
print("# 1. NUDITY — All Ours anchor cells (UD/RAB/MMA/P4DN)")
print("="*100)
NUDITY_DS = [("unlearndiff", "UD"), ("rab", "RAB"), ("ringabell", "RAB"), ("mma", "MMA"), ("p4dn", "P4DN")]

# Search all dirs that look like ours-anchor cells
for ds, label in NUDITY_DS:
    print(f"\n## Dataset: {label} ({ds})")
    candidates = []
    # 1. paper_results_master/01
    for d in glob.glob(f"{BASE}/paper_results_master/01_nudity_sd14_5bench/*{ds}*anchor*"):
        candidates.append(d)
    # 2. launch_0420_nudity all anchor cells for this dataset
    for d in glob.glob(f"{OUT}/launch_0420_nudity/*/{ds}/anchor*"):
        candidates.append(d)
    # 3. main_config family_rab cells (only for rab)
    if ds in ("rab", "ringabell"):
        for d in glob.glob(f"{OUT}/main_config/*ainp*family_rab*"):
            candidates.append(d)
    # 4. launch_0423/0424 anchor cells
    for d in glob.glob(f"{OUT}/launch_042*nudity*/*{ds}*/anchor*"):
        candidates.append(d)
    # 5. older launch dirs
    for d in glob.glob(f"{OUT}/launch_*/{ds}/*anchor*"):
        candidates.append(d)
    candidates = sorted(set(candidates))
    rows = []
    for d in candidates:
        for j in glob.glob(f"{d}/categories_qwen3_vl_nudity*v5*.json"):
            r = sr(j)
            if r is None: continue
            n, s, p, f, nr, srv = r
            args_path = os.path.join(d, "args.json")
            args_summary = ""
            if os.path.exists(args_path):
                try:
                    a = json.load(open(args_path))
                    tc = a.get('target_concepts', [])
                    bug = "[NUDITY-BUG]" if is_nudity_bug(tc) and ds not in ("rab","ringabell","unlearndiff","mma","p4dn") else ""
                    # For nudity datasets, nudity target_concepts is CORRECT
                    args_summary = f" mode={a.get('how_mode','?')} ss={a.get('safety_scale','?')} τ_t={a.get('attn_threshold','?')} τ_i={a.get('img_attn_threshold','?')} τ_cas={a.get('cas_threshold','?')} probe={a.get('probe_mode','?')} fam={os.path.basename(a.get('family_config','?'))} tc={tc}"
                except: pass
            rel = d.replace(OUT+"/", "")
            rows.append((srv, n, rel, args_summary))
    rows.sort(reverse=True, key=lambda x: x[0])
    for srv, n, rel, args in rows[:8]:
        print(f"  SR={srv:5.1f}% n={n}  {rel}")
        if args: print(f"     {args}")

print("\n\n" + "="*100)
print("# 2. I2P — concept-correct alternatives for the 3 nudity-bug cells")
print("="*100)

CONCEPT_KEYWORDS_OK = {
    "self-harm": ["self_harm", "cutting", "pills_suicide", "noose", "self harm", "suicide"],
    "illegal_activity": ["drugs", "crime", "contraband", "illegal", "activity"],
    "harassment": ["bullying", "mockery", "intimidation", "abuse"],
}

for concept, mode_target in [("self-harm", "anchor"), ("illegal_activity", "hybrid"), ("harassment", "hybrid")]:
    print(f"\n## {concept} / {mode_target} — search all cells, list concept-correct ones with v5 SR")
    json_name = {"self-harm": "self_harm", "illegal_activity": "illegal", "harassment": "harassment"}[concept]
    candidates = []
    for d in glob.glob(f"{OUT}/launch_*/**/{concept}/*", recursive=True):
        if os.path.isdir(d): candidates.append(d)
    for d in glob.glob(f"{OUT}/launch_042*/**/i2p_{concept}/*", recursive=True):
        if os.path.isdir(d): candidates.append(d)
    for d in glob.glob(f"{OUT}/launch_*/{concept}/*", recursive=True):
        if os.path.isdir(d): candidates.append(d)
    candidates = sorted(set(candidates))
    rows = []
    for d in candidates:
        # Filter by mode_target hint in dir name
        dname = os.path.basename(d).lower()
        if mode_target == "anchor" and "anchor" not in dname and "ainp" not in dname:
            continue
        if mode_target == "hybrid" and "hybrid" not in dname:
            continue
        # Find v5 json
        jsons = glob.glob(f"{d}/categories_qwen3_vl_{json_name}*v5*.json")
        if not jsons: continue
        # Read args.json
        args_path = os.path.join(d, "args.json")
        if not os.path.exists(args_path): continue
        try:
            a = json.load(open(args_path))
        except: continue
        tc = a.get('target_concepts', [])
        if is_nudity_bug(tc):
            continue  # skip nudity-bug cells
        # Compute SR
        for j in jsons:
            r = sr(j)
            if r is None: continue
            n, s, p, f, nr, srv = r
            rel = d.replace(OUT+"/", "")
            args_summary = f"mode={a.get('how_mode','?')} ss={a.get('safety_scale','?')} τ_t={a.get('attn_threshold','?')} τ_i={a.get('img_attn_threshold','?')} τ_cas={a.get('cas_threshold','?')} probe={a.get('probe_mode','?')} fam={os.path.basename(a.get('family_config','?'))}"
            rows.append((srv, n, rel, args_summary, tc))
    rows.sort(reverse=True, key=lambda x: x[0])
    if not rows:
        print(f"  NO concept-correct {concept}/{mode_target} cells found!")
    for srv, n, rel, args, tc in rows[:8]:
        print(f"  SR={srv:5.1f}% n={n}  {rel}")
        print(f"     {args}")
        print(f"     target_concepts={tc}")
