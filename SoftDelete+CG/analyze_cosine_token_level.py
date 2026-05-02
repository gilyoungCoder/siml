#!/usr/bin/env python
"""
Token-level cosine similarity: for each prompt, compute per-token embeddings
and take the MAX similarity with "nudity" across all tokens.
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
    with open(json_path) as f:
        data = json.load(f)
    cats = {}
    for fname, info in data.items():
        idx = int(fname.replace('.png', ''))
        cats[idx] = info["category"]
    return cats


def compute_word_max_sim(prompts, ref_emb, tokenizer, model, device, batch_size=32):
    """For each prompt, split into words, encode each word individually,
    and take the max cosine sim with 'nudity' across all words."""
    max_sims = []
    top_words = []

    for prompt in prompts:
        # Split prompt into words
        import re
        words = re.findall(r'[a-zA-Z]+', prompt)
        if not words:
            max_sims.append(0.0)
            top_words.append("")
            continue

        # Encode each word individually and get pooler_output
        word_embs = []
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**tokens)
            emb = out.pooler_output.cpu()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            word_embs.append(emb)
        word_embs = torch.cat(word_embs, dim=0)  # (num_words, D)

        # Cosine sim with "nudity"
        sims = (word_embs @ ref_emb.T).squeeze()  # (num_words,)
        if sims.dim() == 0:
            sims = sims.unsqueeze(0)

        max_idx = sims.argmax().item()
        max_sims.append(sims[max_idx].item())
        top_words.append(words[max_idx])

    return torch.tensor(max_sims), top_words


def analyze_dataset(name, prompts, categories, max_sims, top_words):
    n = len(prompts)
    print(f"\n{'='*70}")
    print(f"Dataset: {name} ({n} prompts) - Word-level MAX similarity")
    print(f"{'='*70}")
    print(f"  Mean={max_sims.mean():.4f}, Std={max_sims.std():.4f}, Min={max_sims.min():.4f}, Max={max_sims.max():.4f}")

    if categories:
        cat_counts = {}
        for i in range(n):
            cat = categories.get(i, "Unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"  Baseline categories: {cat_counts}")

        full_set = {i for i in range(n) if categories.get(i) == "Full"}
        partial_set = {i for i in range(n) if categories.get(i) == "Partial"}
        harmful_set = full_set | partial_set
        safe_set = {i for i in range(n) if categories.get(i) in ("Safe", "NotRel")}

        print(f"\n  {'Thr':>6} | {'Above':>6} | {'Full Recall':>12} | {'F+P Recall':>12} | {'Safe skip':>10}")
        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            above = {i for i in range(n) if max_sims[i].item() >= thr}
            full_recall = len(full_set & above) / max(len(full_set), 1) * 100
            harmful_recall = len(harmful_set & above) / max(len(harmful_set), 1) * 100
            safe_skip = len(safe_set - above) / max(len(safe_set), 1) * 100
            print(f"  {thr:>6.2f} | {len(above):>5d} | {full_recall:>10.1f}% | {harmful_recall:>10.1f}% | {safe_skip:>8.1f}%")

        # Show missed Full at threshold 0.5
        thr = 0.50
        missed = sorted([i for i in full_set if max_sims[i].item() < thr])
        if missed:
            print(f"\n  Missed Full (max_word_sim < {thr}):")
            for idx in missed[:15]:
                print(f"    [{idx:03d}] max_sim={max_sims[idx]:.4f} top_word='{top_words[idx]}' | {prompts[idx][:70]}")
            if len(missed) > 15:
                print(f"    ... and {len(missed)-15} more")
    else:
        # COCO style - just show skip rates
        print(f"\n  {'Thr':>6} | {'Below (skip)':>14} | {'Above (guide)':>14}")
        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            below = (max_sims < thr).sum().item()
            above = (max_sims >= thr).sum().item()
            print(f"  {thr:>6.2f} | {below:>5d} ({below/n*100:>5.1f}%) | {above:>5d} ({above/n*100:>5.1f}%)")

    # Show top 10 highest
    print(f"\n  Top 10 highest max_word_sim:")
    top_idx = max_sims.argsort(descending=True)[:10]
    for idx in top_idx:
        cat = categories.get(idx.item(), "?") if categories else "?"
        print(f"    {max_sims[idx]:.4f} [{cat:>7}] top='{top_words[idx]}' | {prompts[idx][:65]}")

    # Show bottom 10
    print(f"\n  Bottom 10 lowest max_word_sim:")
    bot_idx = max_sims.argsort()[:10]
    for idx in bot_idx:
        cat = categories.get(idx.item(), "?") if categories else "?"
        print(f"    {max_sims[idx]:.4f} [{cat:>7}] top='{top_words[idx]}' | {prompts[idx][:65]}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path("/mnt/home/yhgil99/unlearning")
    scg = base / "SoftDelete+CG/scg_outputs"

    clip_model_name = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP: {clip_model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    model = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()

    # "nudity" word embedding (pooler_output, same as each word)
    ref_tokens = tokenizer(["nudity"], return_tensors="pt").to(device)
    with torch.no_grad():
        ref_out = model(**ref_tokens)
    ref_emb = ref_out.pooler_output.cpu()  # (1, D)
    ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)

    # COCO
    print("\nLoading COCO...")
    coco = load_prompts_csv(base / "SAFREE/datasets/coco_30k_10k.csv")
    print(f"  {len(coco)} prompts")
    coco_sims, coco_tokens = compute_word_max_sim(coco, ref_emb, tokenizer, model, device)
    analyze_dataset("COCO", coco, None, coco_sims, coco_tokens)

    # Harmful datasets
    datasets = [
        ("Ring-A-Bell", base / "SAFREE/datasets/nudity-ring-a-bell.csv", scg / "final_ringabell/sd_baseline/categories_qwen3_vl_nudity.json", "csv"),
        ("UnlearnDiff", base / "SAFREE/datasets/unlearn_diff_nudity.csv", scg / "final_unlearndiff/sd_baseline/categories_qwen3_vl_nudity.json", "csv"),
        ("MMA", base / "prompts/nudity_datasets/mma.txt", scg / "final_mma/sd_baseline/categories_qwen3_vl_nudity.json", "txt"),
    ]

    for name, prompt_path, cat_path, loader in datasets:
        print(f"\nLoading {name}...")
        prompts = load_prompts_csv(prompt_path) if loader == "csv" else load_prompts_txt(prompt_path)
        categories = load_categories(cat_path)
        n = min(len(prompts), len(categories))
        prompts = prompts[:n]
        print(f"  {len(prompts)} prompts")

        sims, tokens = compute_word_max_sim(prompts, ref_emb, tokenizer, model, device)
        analyze_dataset(name, prompts, categories, sims, tokens)


if __name__ == "__main__":
    main()
