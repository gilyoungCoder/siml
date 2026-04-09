#!/usr/bin/env python3
"""
결과 집계
"""
import os
import re
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.path_utils import get_outputs_root
from vlm.result_contract import parse_nudenet_result_file, parse_category_report_file

BASE = str(get_outputs_root())

DATASETS = ["i2p", "mma", "ringabell"]
DATASET_NAMES = {
    "i2p": "I2P",
    "mma": "MMA",
    "ringabell": "Ring-A-Bell"
}

MODELS = ["baseline", "ours_4class", "safree_ours_4class", "safree_regenerated"]
MODEL_NAMES = {
    "baseline": "SD-v1.4",
    "ours_4class": "Ours",
    "safree_ours_4class": "SAFREE+Ours",
    "safree_regenerated": "SAFREE"
}

def parse_nudenet(path):
    """NudeNet results_nudenet.txt 파싱 -> Unsafe Rate"""
    parsed = parse_nudenet_result_file(path)
    return parsed['unsafe_rate'] if parsed else None

def parse_gpt4o_detail(path):
    """GPT-4o results_gpt4o.txt 파싱 -> dict with Safe, Partial, Full, NotPeople, SR"""
    summary = parse_category_report_file(path)
    if summary is None or summary['total'] == 0:
        return None
    counts = summary['counts']
    total = summary['total']
    return {
        'safe': counts['Safe'] / total,
        'partial': counts['Partial'] / total,
        'full': counts['Full'] / total,
        'notrel': counts['NotRel'] / total,
        'sr': summary['sr'],
    }

def get_result_dir(dataset, model):
    """결과 파일 디렉토리 반환"""
    base = f"{BASE}/nudity_datasets/{dataset}/{model}"
    if model == "safree_regenerated":
        return os.path.join(base, "generated")
    return base

def parse_coco_metrics(model):
    """COCO eval_metrics.json 파싱"""
    path = f"{BASE}/coco/{model}/eval_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# ============================================
# Table 1: Main Table (SR / NudeNet + COCO)
# ============================================
print("=" * 95)
print("Table 1: Main Results")
print("Nudity: SR (↑) / NudeNet (↓)  |  COCO: FID (↓) / CLIP (↑)")
print("=" * 95)
print()

header = f"{'Method':<15}"
for dataset in DATASETS:
    header += f" | {DATASET_NAMES[dataset]:^15}"
header += f" |       COCO      "
print(header)

subheader = f"{'':<15}"
for _ in DATASETS:
    subheader += f" | {'(SR / NudeNet)':^15}"
subheader += f" | {'(FID / CLIP)':^15}"
print(subheader)
print("-" * 95)

for model in MODELS:
    row = f"{MODEL_NAMES[model]:<15}"

    # Nudity datasets
    for dataset in DATASETS:
        result_dir = get_result_dir(dataset, model)
        gpt = parse_gpt4o_detail(os.path.join(result_dir, "results_gpt4o.txt"))
        nn = parse_nudenet(os.path.join(result_dir, "results_nudenet.txt"))

        sr = gpt['sr'] if gpt else None
        if sr is not None and nn is not None:
            cell = f"{sr:.2f} / {nn:.2f}"
        else:
            cell = "N/A"
        row += f" | {cell:^15}"

    # COCO
    coco = parse_coco_metrics(model)
    if coco:
        coco_cell = f"{coco['fid']:.1f} / {coco['clip_score']:.3f}"
    else:
        coco_cell = "-"
    row += f" | {coco_cell:^15}"

    print(row)

print("-" * 95)


# ============================================
# Table 2: Detailed breakdown per dataset
# ============================================
for dataset in DATASETS:
    print()
    print("=" * 85)
    print(f"Table 2-{dataset}: GPT-4o Detailed Results on {DATASET_NAMES[dataset]}")
    print("SR = Safe + Partial. All values are proportions.")
    print("=" * 85)
    print()

    header = f"{'Method':<15} | {'Safe':^6} | {'Partial':^7} | {'Full':^6} | {'NotRel':^6} | {'SR (↑)':^8} | {'NudeNet (↓)':^11}"
    print(header)
    print("-" * 85)

    for model in MODELS:
        result_dir = get_result_dir(dataset, model)
        gpt = parse_gpt4o_detail(os.path.join(result_dir, "results_gpt4o.txt"))
        nn = parse_nudenet(os.path.join(result_dir, "results_nudenet.txt"))

        if gpt:
            row = f"{MODEL_NAMES[model]:<15} | {gpt['safe']:^6.2f} | {gpt['partial']:^7.2f} | {gpt['full']:^6.2f} | {gpt['notrel']:^6.2f} | {gpt['sr']:^8.2f} | {nn:^11.2f}" if nn else f"{MODEL_NAMES[model]:<15} | {gpt['safe']:^6.2f} | {gpt['partial']:^7.2f} | {gpt['full']:^6.2f} | {gpt['notrel']:^6.2f} | {gpt['sr']:^8.2f} | {'N/A':^11}"
        else:
            row = f"{MODEL_NAMES[model]:<15} | {'N/A':^6} | {'N/A':^7} | {'N/A':^6} | {'N/A':^6} | {'N/A':^8} | {nn:^11.2f}" if nn else f"{MODEL_NAMES[model]:<15} | {'N/A':^6} | {'N/A':^7} | {'N/A':^6} | {'N/A':^6} | {'N/A':^8} | {'N/A':^11}"
        print(row)

    print("-" * 85)
