import json, os, glob, sys

base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_clean"
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
        sr = 100 * safe / total
        nr = 100 * cats.get("NotRel", 0) / total
        full = 100 * cats.get("Full", 0) / total
        results.append((sr, d, concept, nr, full, total))

results.sort(reverse=True)

# Nudity results
nudity = [r for r in results if r[2] == "nudity"]
concepts = [r for r in results if r[2] != "nudity"]

print("=" * 75)
print("  v27 NUDITY RESULTS (Ring-A-Bell, Qwen3-VL)")
print("=" * 75)
print("{:50s} {:>6s} {:>6s} {:>6s}".format("Config", "SR%", "NR%", "Full%"))
print("-" * 75)
for sr, d, concept, nr, full, n in nudity[:40]:
    m = " <<<" if sr >= 92 else (" **" if sr >= 90 else "")
    print("{:50s} {:5.1f}% {:5.1f}% {:5.1f}%{}".format(d[:50], sr, nr, full, m))
print("...(total {})".format(len(nudity)))

if concepts:
    print()
    print("=" * 75)
    print("  v27 MULTI-CONCEPT RESULTS")
    print("=" * 75)
    print("{:50s} {:10s} {:>6s} {:>6s} {:>6s}".format("Config", "Concept", "SR%", "NR%", "Full%"))
    print("-" * 80)
    for sr, d, concept, nr, full, n in sorted(concepts, key=lambda x: (x[2], -x[0])):
        print("{:50s} {:10s} {:5.1f}% {:5.1f}% {:5.1f}%".format(d[:50], concept, sr, nr, full))
