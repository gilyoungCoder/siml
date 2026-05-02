#!/usr/bin/env python
"""
Compute CLIP text embedding cosine similarity between prompts and "nudity".
Compare COCO (safe) vs Ring-A-Bell + UnlearnDiff + MMA (harmful).
"""

import csv
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
            raise ValueError(f"No prompt column found in {path}. Available: {reader.fieldnames}")
        print(f"  [{path.name}] Using column '{col}'")
        for row in reader:
            p = row[col].strip()
            if p:
                prompts.append(p)
    return prompts


def compute_embeddings(prompts, tokenizer, model, device, batch_size=64):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tokens)
        # Use pooled output (EOS token embedding)
        emb = out.pooler_output  # (B, 768)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP text model
    clip_model_name = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP text model: {clip_model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    model = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()

    # Reference word embedding
    ref_tokens = tokenizer(["nudity"], return_tensors="pt").to(device)
    with torch.no_grad():
        ref_emb = model(**ref_tokens).pooler_output
    ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
    ref_emb = ref_emb.cpu()

    # Load datasets
    base = Path("/mnt/home/yhgil99/unlearning")

    print("\n=== Loading COCO (safe) ===")
    coco = load_prompts_csv(base / "SAFREE/datasets/coco_30k_10k.csv")
    print(f"  {len(coco)} prompts")

    print("\n=== Loading harmful datasets ===")
    ringabell = load_prompts_csv(base / "SAFREE/datasets/nudity-ring-a-bell.csv")
    print(f"  Ring-A-Bell: {len(ringabell)} prompts")

    unlearndiff = load_prompts_csv(base / "SAFREE/datasets/unlearn_diff_nudity.csv")
    print(f"  UnlearnDiff: {len(unlearndiff)} prompts")

    mma = load_prompts_txt(base / "prompts/nudity_datasets/mma.txt")
    print(f"  MMA: {len(mma)} prompts")

    harmful_all = ringabell + unlearndiff + mma
    print(f"\n  Total harmful: {len(harmful_all)} prompts")

    # Compute embeddings
    print("\nComputing COCO embeddings...")
    coco_emb = compute_embeddings(coco, tokenizer, model, device)

    print("Computing harmful embeddings...")
    harmful_emb = compute_embeddings(harmful_all, tokenizer, model, device)

    # Cosine similarity with "nudity"
    coco_sim = (coco_emb @ ref_emb.T).squeeze()
    harmful_sim = (harmful_emb @ ref_emb.T).squeeze()

    # Also compute per-dataset
    n_ring = len(ringabell)
    n_ud = len(unlearndiff)
    n_mma = len(mma)
    ring_sim = harmful_sim[:n_ring]
    ud_sim = harmful_sim[n_ring:n_ring+n_ud]
    mma_sim = harmful_sim[n_ring+n_ud:]

    print(f"\n{'='*60}")
    print(f"Cosine Similarity with 'nudity'")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    print(f"{'-'*60}")

    for name, sim in [
        ("COCO (safe)", coco_sim),
        ("Ring-A-Bell", ring_sim),
        ("UnlearnDiff", ud_sim),
        ("MMA", mma_sim),
        ("Harmful (all)", harmful_sim),
    ]:
        print(f"{name:<20} {sim.mean():>8.4f} {sim.std():>8.4f} {sim.min():>8.4f} {sim.max():>8.4f} {sim.median():>8.4f}")

    # Distribution overlap analysis
    print(f"\n{'='*60}")
    print(f"Threshold Analysis")
    print(f"{'='*60}")
    for thr in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        coco_above = (coco_sim >= thr).float().mean().item() * 100
        harmful_above = (harmful_sim >= thr).float().mean().item() * 100
        print(f"  thr={thr:.2f}: COCO above={coco_above:5.1f}% | Harmful above={harmful_above:5.1f}%")

    # Show some examples
    print(f"\n{'='*60}")
    print("Top 10 COCO prompts (highest sim with 'nudity'):")
    top_coco_idx = coco_sim.argsort(descending=True)[:10]
    for idx in top_coco_idx:
        print(f"  {coco_sim[idx]:.4f} | {coco[idx][:80]}")

    print(f"\nBottom 10 harmful prompts (lowest sim with 'nudity'):")
    bot_harm_idx = harmful_sim.argsort()[:10]
    for idx in bot_harm_idx:
        print(f"  {harmful_sim[idx]:.4f} | {harmful_all[idx][:80]}")


if __name__ == "__main__":
    main()
