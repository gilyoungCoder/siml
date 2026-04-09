import json, os

bases = [
    "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v24_stage2",
    "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v24_stage3",
    "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v24",
]
results = []
seen = set()
for base in bases:
    if not os.path.exists(base): continue
    for d in sorted(os.listdir(base)):
        if d in seen or d.startswith("test"): continue
        seen.add(d)
        rf = os.path.join(base, d, "categories_qwen3_vl_nudity.json")
        if not os.path.exists(rf): continue
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
        results.append((sr, d, nr, full, total))

results.sort(reverse=True)
print("Total evaluated:", len(results))
print()
hdr = "{:50s} {:>6s} {:>6s} {:>6s} {:>5s}".format("Config", "SR%", "NR%", "Full%", "n")
print(hdr)
print("-" * 75)
for sr, d, nr, full, n in results[:30]:
    m = " <<<" if sr >= 92 else (" **" if sr >= 90 else "")
    print("{:50s} {:5.1f}% {:5.1f}% {:5.1f}% {:5d}{}".format(d, sr, nr, full, n, m))
