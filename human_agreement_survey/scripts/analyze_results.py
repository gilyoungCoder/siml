#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, itertools
from collections import Counter, defaultdict
from pathlib import Path

ORDER = {"NotRelevant":0, "Safe":1, "Partial":2, "Full":3}
CONCEPTS = ["sexual","violence","self-harm","hate","shocking","harassment","illegal_activity","disturbing"]

def majority(labels):
    c = Counter(labels)
    if not c: return None
    # deterministic tie-break toward median severity after frequency
    return sorted(c.items(), key=lambda kv: (-kv[1], ORDER.get(kv[0], 99)))[0][0]

def main():
    ap = argparse.ArgumentParser(description="Compute Table 16 human-majority agreement from Supabase CSV export.")
    ap.add_argument("csv", help="CSV exported from Supabase human_agreement_annotations")
    ap.add_argument("--qwen", default="data/qwen_labels_private.json")
    ap.add_argument("--min-votes", type=int, default=3)
    args = ap.parse_args()
    qwen = json.load(open(args.qwen))
    votes = defaultdict(list)
    annotators = set()
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("label") in ORDER and r.get("item_id"):
                votes[r["item_id"]].append(r["label"])
                annotators.add(r.get("annotator_code", ""))
    rows=[]
    for item_id, labs in votes.items():
        if len(labs) < args.min_votes or item_id not in qwen: continue
        maj = majority(labs); q = qwen[item_id]["qwen_label"]; concept=qwen[item_id]["concept"]
        rows.append({"item_id":item_id,"concept":concept,"n":len(labs),"majority":maj,"qwen":q,"exact":q==maj,"within1":abs(ORDER[q]-ORDER[maj])<=1})
    def agg(rs):
        if not rs: return None
        return sum(r["exact"] for r in rs)/len(rs)*100, sum(r["within1"] for r in rs)/len(rs)*100, len(rs)
    print(f"Annotators: {len(annotators)} | items with >={args.min_votes} votes and Qwen key: {len(rows)}")
    print("\nLaTeX row candidates (Exact / Within-1):")
    all_rs=[]
    for c in CONCEPTS:
        rs=[r for r in rows if r["concept"]==c]
        a=agg(rs)
        if a:
            print(f"{c.replace(_,-)} & {a[0]:.1f} & {a[1]:.1f} \\  % n={a[2]}")
            all_rs.extend(rs)
        else:
            print(f"{c.replace(_,-)} & TBD & TBD \\  % n=0")
    a=agg(all_rs)
    if a: print(f"Pooled & {a[0]:.1f} & {a[1]:.1f} \\  % n={a[2]}")
    out=Path("data/human_agreement_item_results.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
if __name__ == "__main__": main()
