#!/usr/bin/env python3
"""Collect bundled or generated Qwen3-VL v5 result text files into a CSV table."""
import argparse
import csv
import re
from pathlib import Path

LABELS = ["Safe", "Partial", "Full", "NotRelevant"]
ALIASES = {"NotRelevant": ["NotRelevant", "NotRel", "NotPeople", "NR"]}

def pct(text, label):
    for name in ALIASES.get(label, [label]):
        m = re.search(rf"{re.escape(name)}:\s*\d+\s*\(([-0-9.]+)%\)", text)
        if m:
            return float(m.group(1))
    return None

def concept_from_file(path):
    name = path.name
    m = re.search(r"qwen3_vl_([^_]+(?:_[^_]+)?)_v5", name)
    if m:
        return m.group(1)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Directory to scan recursively.")
    ap.add_argument("--out", default=None, help="CSV output path; default stdout.")
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.root).rglob("*v5.txt")):
        text = f.read_text(errors="ignore")
        if "Distribution:" not in text and "SR (Safe+Partial)" not in text:
            continue
        vals = {lab: pct(text, lab) for lab in LABELS}
        safe = vals.get("Safe") or 0.0
        partial = vals.get("Partial") or 0.0
        vals["SR"] = round(safe + partial, 1)
        rows.append({"concept_or_rubric": concept_from_file(f), "result_file": str(f), "outdir": str(f.parent), **vals})
    fieldnames = ["concept_or_rubric", "result_file", "outdir", "SR", "Safe", "Partial", "Full", "NotRelevant"]
    fp = open(args.out, "w", newline="") if args.out else __import__("sys").stdout
    with fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
