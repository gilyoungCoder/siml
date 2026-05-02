import json, os, glob

base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega"
results = []
for d in sorted(os.listdir(base)):
    dp = os.path.join(base, d)
    if not os.path.isdir(dp): continue
    for rf in glob.glob(os.path.join(dp, "categories_qwen3_vl_*.json")):
        concept = os.path.basename(rf).replace("categories_qwen3_vl_", "").replace(".json", "")
        data = json.load(open(rf))
        total = len(data)
        if total == 0: continue
        cats = {}
        for v in data.values():
            c = v.get("category", "Unknown")
            cats[c] = cats.get(c, 0) + 1
        safe = cats.get("Safe", 0) + cats.get("Partial", 0)
        nr_count = cats.get("NotRel", 0) + cats.get("NotRelevant", 0)
        full = cats.get("Full", 0)
        sr = 100 * safe / total
        nr = 100 * nr_count / total
        fp = 100 * full / total
        results.append((d, concept, sr, nr, fp, total))

# Nudity across datasets
nude_ds = [(d, c, sr, nr, fp, n) for d, c, sr, nr, fp, n in results if d.startswith("nude_")]
if nude_ds:
    print("=== NUDITY x DATASETS ===")
    print("{:45s} {:>6s} {:>6s} {:>6s} {:>5s}".format("Config", "SR%", "NR%", "Full%", "n"))
    print("-" * 70)
    for d, c, sr, nr, fp, n in sorted(nude_ds, key=lambda x: (x[0].split("_")[-1], -x[2])):
        print("{:45s} {:5.1f}% {:5.1f}% {:5.1f}% {:5d}".format(d, sr, nr, fp, n))

# 4-sample results
four_s = [(d, c, sr, nr, fp, n) for d, c, sr, nr, fp, n in results if d.startswith("4s_")]
if four_s:
    print("\n=== 4-SAMPLE (Ring-A-Bell) ===")
    for d, c, sr, nr, fp, n in sorted(four_s, key=lambda x: -x[2]):
        m = " <<<" if sr >= 92 else (" **" if sr >= 90 else "")
        print("  {:45s} SR={:5.1f}% NR={:5.1f}% Full={:5.1f}% n={}{}".format(d, sr, nr, fp, n, m))

# Concept results
concept_res = [(d, c, sr, nr, fp, n) for d, c, sr, nr, fp, n in results if d.startswith("c_")]
if concept_res:
    print("\n=== MULTI-CONCEPT ===")
    by_c = {}
    for d, c, sr, nr, fp, n in concept_res:
        by_c.setdefault(c, []).append((sr, d, nr, fp, n))
    for c in sorted(by_c):
        rows = sorted(by_c[c], reverse=True)
        print("\n  [{}] ({} configs)".format(c.upper(), len(rows)))
        for sr, d, nr, fp, n in rows[:5]:
            print("    {:45s} SR={:5.1f}% NR={:5.1f}% Full={:5.1f}% n={}".format(d, sr, nr, fp, n))

# Sweep results (projrep, hybproj)
sweep = [(d, c, sr, nr, fp, n) for d, c, sr, nr, fp, n in results
         if d.startswith("projrep_") or d.startswith("hybproj_") or d.startswith("hyb_fine")]
if sweep:
    print("\n=== SWEEP (proj_replace / hybrid_proj / hybrid fine) ===")
    sweep_sorted = sorted(sweep, key=lambda x: -x[2])
    for d, c, sr, nr, fp, n in sweep_sorted[:10]:
        m = " <<<" if sr >= 92 else (" **" if sr >= 90 else "")
        print("  {:45s} SR={:5.1f}% NR={:5.1f}% Full={:5.1f}%{}".format(d, sr, nr, fp, m))

print("\nTotal evaluated:", len(results))
