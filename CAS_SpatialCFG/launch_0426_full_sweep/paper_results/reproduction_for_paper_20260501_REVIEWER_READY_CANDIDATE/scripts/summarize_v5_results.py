#!/usr/bin/env python3
"""Collect Qwen3-VL v5 result text files into a CSV table."""
import argparse
import csv
import re
from pathlib import Path

LABELS = ["Safe", "Partial", "Full", "NotRelevant"]
ALIASES = {"NotRelevant": ["NotRelevant", "NotRel", "NotPeople", "NR"]}

def pct(text, label):
    names = ALIASES.get(label, [label])
    for name in names:
        m = re.search(rf"{re.escape(name)}:\s*\d+\s*\(([-0-9.]+)%\)", text)
        if m:
            return float(m.group(1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Directory to scan recursively.")
    ap.add_argument("--out", default=None, help="CSV output path; default stdout.")
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.root).rglob("results_qwen3_vl_*_v5.txt")):
        text = f.read_text(errors="ignore")
        vals = {lab: pct(text, lab) for lab in LABELS}
        safe = vals.get("Safe") or 0.0
        partial = vals.get("Partial") or 0.0
        vals["SR"] = round(safe + partial, 1)
        rows.append({"result_file": str(f), "outdir": str(f.parent), **vals})
    fieldnames = ["result_file", "outdir", "SR", "Safe", "Partial", "Full", "NotRelevant"]
    if args.out:
        fp = open(args.out, "w", newline="")
    else:
        import sys
        fp = sys.stdout
    with fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
