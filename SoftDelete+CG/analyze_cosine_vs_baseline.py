#!/usr/bin/env python
"""
Compare CLIP cosine similarity threshold vs actual SD baseline nudity detection.
For each dataset: check overlap between cos_sim >= threshold and baseline "Full" category.
"""

import csv
import json
import numpy as np
import torch
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel


def load_prompts_txt(path):
    return [l.strip() for l in open(path) if l.strip()]


def load_prompts_csv(path, column_priority=None):
    if column_priority is None:
        column_priority = ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text']
    prompts = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        col = None
        for c in column_priority:
            if c in reader.fieldnames:
                col = c
                break
        if col is None:
            raise ValueError(f"No prompt column found. Available: {reader.fieldnames}")
        print(f"  [{Path(path).name}] Using column '{col}'")
        for row in reader:
            p = row[col].strip()
            if p:
                prompts.append(p)
    return prompts


def load_categories(json_path):
    """Load categories, return dict mapping index -> category."""
    with open(json_path) as f:
        data = json.load(f)
    cats = {}
    for fname, info in data.items():
        # Extract index from filename like "000042.png"
        idx = int(fname.replace('.png', ''))
        cats[idx] = info["category"]
    return cats


def compute_embeddings(prompts, tokenizer, model, device, batch_size=64):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tokens)
        emb = out.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def analyze_dataset(name, prompts, categories, sims, threshold=0.3):
    """Analyze overlap between cos_sim >= threshold and baseline Full category."""
    n = len(prompts)
    print(f"\n{'='*70}")
    print(f"Dataset: {name} ({n} prompts)")
    print(f"{'='*70}")

    # Count categories
    cat_counts = {}
    for i in range(n):
        cat = categories.get(i, "Unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"Baseline categories: {cat_counts}")

    # Full nudity in baseline
    full_set = {i for i in range(n) if categories.get(i) == "Full"}
    partial_set = {i for i in range(n) if categories.get(i) == "Partial"}
    harmful_set = full_set | partial_set  # Full + Partial = harmful

    # Above threshold
    above_thr = {i for i in range(n) if sims[i].item() >= threshold}

    # Overlap analysis
    print(f"\nThreshold: {threshold}")
    print(f"  Above threshold (cos_sim >= {threshold}): {len(above_thr)}/{n} ({len(above_thr)/n*100:.1f}%)")
    print(f"  Baseline Full: {len(full_set)}/{n} ({len(full_set)/n*100:.1f}%)")
    print(f"  Baseline Full+Partial: {len(harmful_set)}/{n} ({len(harmful_set)/n*100:.1f}%)")

    # How many "Full" are caught by threshold?
    full_caught = full_set & above_thr
    full_missed = full_set - above_thr
    print(f"\n  [Recall] Full caught by threshold: {len(full_caught)}/{len(full_set)} ({len(full_caught)/max(len(full_set),1)*100:.1f}%)")
    print(f"  [Recall] Full+Partial caught: {len(harmful_set & above_thr)}/{len(harmful_set)} ({len(harmful_set & above_thr)/max(len(harmful_set),1)*100:.1f}%)")

    # False negatives: Full but below threshold
    if full_missed:
        print(f"\n  Missed Full prompts (below threshold):")
        missed_list = sorted(full_missed)
        for idx in missed_list[:15]:
            print(f"    [{idx:03d}] sim={sims[idx]:.4f} | {prompts[idx][:80]}")
        if len(missed_list) > 15:
            print(f"    ... and {len(missed_list)-15} more")

    # Multi-threshold sweep
    print(f"\n  Threshold sweep:")
    print(f"  {'Thr':>6} | {'Above':>6} | {'Full Recall':>12} | {'F+P Recall':>12} | {'Safe skip':>10}")
    safe_set = {i for i in range(n) if categories.get(i) in ("Safe", "NotRel")}
    for thr in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]:
        above = {i for i in range(n) if sims[i].item() >= thr}
        full_recall = len(full_set & above) / max(len(full_set), 1) * 100
        harmful_recall = len(harmful_set & above) / max(len(harmful_set), 1) * 100
        safe_skip = len(safe_set - above) / max(len(safe_set), 1) * 100
        print(f"  {thr:>6.2f} | {len(above):>5d} | {full_recall:>10.1f}% | {harmful_recall:>10.1f}% | {safe_skip:>8.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path("/mnt/home/yhgil99/unlearning")
    scg = base / "SoftDelete+CG/scg_outputs"

    # Load CLIP
    clip_model_name = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP: {clip_model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    model = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()

    # Reference embedding
    ref_tokens = tokenizer(["nudity"], return_tensors="pt").to(device)
    with torch.no_grad():
        ref_emb = model(**ref_tokens).pooler_output
    ref_emb = (ref_emb / ref_emb.norm(dim=-1, keepdim=True)).cpu()

    # Datasets config
    datasets = [
        {
            "name": "Ring-A-Bell",
            "prompts_path": base / "SAFREE/datasets/nudity-ring-a-bell.csv",
            "categories_path": scg / "final_ringabell/sd_baseline/categories_qwen3_vl_nudity.json",
            "loader": "csv",
        },
        {
            "name": "UnlearnDiff",
            "prompts_path": base / "SAFREE/datasets/unlearn_diff_nudity.csv",
            "categories_path": scg / "final_unlearndiff/sd_baseline/categories_qwen3_vl_nudity.json",
            "loader": "csv",
        },
        {
            "name": "MMA",
            "prompts_path": base / "prompts/nudity_datasets/mma.txt",
            "categories_path": scg / "final_mma/sd_baseline/categories_qwen3_vl_nudity.json",
            "loader": "txt",
        },
    ]

    # COCO skip analysis (no baseline categories needed)
    print(f"\n{'='*70}")
    print(f"COCO Skip Analysis (cosine sim with 'nudity')")
    print(f"{'='*70}")
    coco = load_prompts_csv(base / "SAFREE/datasets/coco_30k_10k.csv")
    print(f"  {len(coco)} prompts")
    coco_emb = compute_embeddings(coco, tokenizer, model, device)
    coco_sims = (coco_emb @ ref_emb.T).squeeze()
    print(f"  Mean={coco_sims.mean():.4f}, Std={coco_sims.std():.4f}, Min={coco_sims.min():.4f}, Max={coco_sims.max():.4f}")
    print(f"\n  {'Thr':>6} | {'Below (skip)':>14} | {'Above (guide)':>14}")
    for thr in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]:
        below = (coco_sims < thr).sum().item()
        above = (coco_sims >= thr).sum().item()
        print(f"  {thr:>6.2f} | {below:>5d} ({below/len(coco)*100:>5.1f}%) | {above:>5d} ({above/len(coco)*100:>5.1f}%)")

    print(f"\n  Top 20 COCO prompts (highest sim):")
    top_idx = coco_sims.argsort(descending=True)[:20]
    for idx in top_idx:
        print(f"    {coco_sims[idx]:.4f} | {coco[idx][:80]}")
    print()

    for ds in datasets:
        print(f"\nLoading {ds['name']}...")
        if ds["loader"] == "csv":
            prompts = load_prompts_csv(ds["prompts_path"])
        else:
            prompts = load_prompts_txt(ds["prompts_path"])
        print(f"  {len(prompts)} prompts")

        categories = load_categories(ds["categories_path"])
        print(f"  {len(categories)} category entries")

        # Trim to min of both
        n = min(len(prompts), len(categories))
        prompts = prompts[:n]

        emb = compute_embeddings(prompts, tokenizer, model, device)
        sims = (emb @ ref_emb.T).squeeze()

        analyze_dataset(ds["name"], prompts, categories, sims, threshold=0.3)


if __name__ == "__main__":
    main()
