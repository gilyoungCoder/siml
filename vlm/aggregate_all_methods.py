#!/usr/bin/env python3
"""
Aggregate all methods comparison table.
Methods: SD Baseline, ESD, SAFREE, SAFREE+Ours, Ours (mon4class)
Datasets: i2p, mma, p4dn, ringabell, unlearndiff
"""
import json
import os
import sys
import argparse

from result_contract import load_category_json_summary
from result_paths import categories_json_candidates, find_existing_result_file
from path_utils import get_scg_outputs_root

def calc_sr(json_path):
    """Calculate canonical SR = (Safe + Partial) / Total * 100."""
    resolved = find_existing_result_file(
        os.path.dirname(json_path),
        categories_json_candidates("qwen", "nudity"),
    )
    if resolved is None:
        return None, None
    summary = load_category_json_summary(resolved)
    if summary["total"] == 0:
        return None, None
    sr = summary["sr"] * 100
    return sr, {"total": summary["total"], **summary["counts"]}

BASE = str(get_scg_outputs_root())
DATASETS = ["mma", "p4dn", "ringabell", "unlearndiff"]

# Method definitions: name -> {dataset: path_to_json}
METHODS = {
    "SD Baseline": {ds: f"{BASE}/final_{ds}/sd_baseline/categories_qwen_nudity.json" for ds in DATASETS},
    "ESD": {ds: f"{BASE}/final_{ds}/esd/categories_qwen_nudity.json" for ds in DATASETS},
    "SDD": {ds: f"{BASE}/final_{ds}/sdd/categories_qwen_nudity.json" for ds in DATASETS},
    "SAFREE": {ds: f"{BASE}/final_{ds}/safree/categories_qwen_nudity.json" for ds in DATASETS},
    "SAFREE+Ours": {ds: f"{BASE}/final_{ds}/safree_mon/mon0.2_gs5_bs2.0_sp0.7-0.3/categories_qwen_nudity.json" for ds in DATASETS},
    "Ours": {ds: f"{BASE}/fine_grid_mon4class/{ds}/mon0.05_gs12.5_bs2.0_sp0.2-0.3/categories_qwen_nudity.json" for ds in DATASETS},
    "SAFREE+Ours_txtskip": {ds: f"{BASE}/final_{ds}/safree_ours_text_exit/categories_qwen_nudity.json" for ds in DATASETS},
    "Ours_txtskip": {ds: f"{BASE}/text_exit_20260202_184334/{ds}/mon0.05_gs12.5_bs2.0_sp0.2-0.3_txt0.50/categories_qwen_nudity.json" for ds in DATASETS},
}

# COCO FID: method -> path to eval_metrics.json
COCO_FID = {
    "SD Baseline": f"{BASE}/final_coco/sd_baseline/eval_metrics.json",
    "ESD": f"{BASE}/final_coco/esd/eval_metrics.json",
    "SDD": f"{BASE}/final_coco/sdd/eval_metrics.json",
    "SAFREE": f"{BASE}/final_coco/safree/eval_metrics.json",
    "SAFREE+Ours": f"{BASE}/final_coco/safree_ours/eval_metrics.json",
    "Ours": f"{BASE}/final_coco/ours/eval_metrics.json",
    "SAFREE+Ours_txtskip": f"{BASE}/final_coco/safree_ours_text_exit/eval_metrics.json",
    "Ours_txtskip": f"{BASE}/text_exit_20260202_184334/coco/mon0.05_gs12.5_bs2.0_sp0.2-0.3_txt0.50/eval_metrics.json",
}

def get_coco_metrics(method_name):
    """Return (fid, clip_score) from eval_metrics.json."""
    path = COCO_FID.get(method_name, "")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get("fid"), data.get("clip_score")
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-i2p", action="store_true", help="Exclude i2p dataset")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    ds_list = [d for d in DATASETS if not (args.no_i2p and d == "i2p")]

    # Header
    if args.latex:
        cols = " & ".join(ds_list) + " & AVG & FID & CLIP"
        print(f"Method & {cols} \\\\")
        print("\\hline")
    else:
        header = f"{'Method':<20}"
        for ds in ds_list:
            header += f"{ds:>12}"
        header += f"{'AVG':>8}{'FID':>8}{'CLIP':>8}"
        print(header)
        print("-" * len(header))

    for method_name, paths in METHODS.items():
        srs = {}
        for ds in ds_list:
            sr, detail = calc_sr(paths[ds])
            srs[ds] = sr

        valid = [v for v in srs.values() if v is not None]
        avg = sum(valid) / len(valid) if valid else None
        fid, clip_score = get_coco_metrics(method_name)

        if args.latex:
            vals = []
            for ds in ds_list:
                if srs[ds] is not None:
                    vals.append(f"{srs[ds]:.1f}\\%")
                else:
                    vals.append("N/A")
            avg_str = f"{avg:.1f}\\%" if avg else "N/A"
            fid_str = f"{fid:.2f}" if fid else "N/A"
            clip_str = f"{clip_score:.4f}" if clip_score else "N/A"
            print(f"{method_name} & {' & '.join(vals)} & {avg_str} & {fid_str} & {clip_str} \\\\")
        else:
            row = f"{method_name:<20}"
            for ds in ds_list:
                if srs[ds] is not None:
                    row += f"{srs[ds]:>11.1f}%"
                else:
                    row += f"{'N/A':>12}"
            if avg is not None:
                row += f"{avg:>7.1f}%"
            else:
                row += f"{'N/A':>8}"
            if fid is not None:
                row += f"{fid:>8.2f}"
            else:
                row += f"{'N/A':>8}"
            if clip_score is not None:
                row += f"{clip_score:>8.4f}"
            else:
                row += f"{'N/A':>8}"
            print(row)

    # Detailed breakdown
    if not args.latex:
        print("\n\n=== Detailed Breakdown ===")
        for method_name, paths in METHODS.items():
            print(f"\n--- {method_name} ---")
            for ds in ds_list:
                sr, detail = calc_sr(paths[ds])
                if sr is not None:
                    print(f"  {ds:<12} SR={sr:5.1f}%  (n={detail['total']})  "
                          f"NotRel:{detail['NotRel']} | Safe:{detail['Safe']} | "
                          f"Partial:{detail['Partial']} | Full:{detail['Full']}")
                else:
                    print(f"  {ds:<12} N/A")

if __name__ == "__main__":
    main()
