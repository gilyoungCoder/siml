import json
hits = json.load(open("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/triple_hits.json"))
picks = [8, 23, 26, 30, 32, 48, 109]
chosen = [hits[i-1] for i in picks]
for n,r in zip(picks, chosen):
    print("#%3d: %-22s SAF=%s" % (n, r["tag"], r["saf_label"]))
json.dump(chosen, open("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/figures_qual/final_picks.json","w"), indent=2)
