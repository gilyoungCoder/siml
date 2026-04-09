#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v26 Embedding Prep: CLIP image encoder intermediate hidden states + visual_projection

SD1.4 text encoder = 12 layers, uses output after ALL 12 layers.
CLIP image encoder = 24 layers.
We extract at layer 12 (50%, same "depth ratio") and layer 24 (last).
Then project 1024 → 768 using CLIP's own visual_projection.

Usage:
  python prepare_clip_v26.py --concept nudity
  python prepare_clip_v26.py --all
"""

import torch, torch.nn.functional as F
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).parent.parent
EXEMPLAR_DIR = REPO / "CAS_SpatialCFG" / "exemplars"


def load_images(img_dir, prefix, max_n=32):
    imgs = []
    for f in sorted(Path(img_dir).iterdir()):
        if f.name.startswith(prefix) and f.suffix == ".png":
            imgs.append(Image.open(f).convert("RGB"))
            if len(imgs) >= max_n:
                break
    return imgs


def extract_hidden_states(images, device, layers=[12, 24]):
    """Extract hidden states at specified layers from CLIP ViT-L/14."""
    from transformers import CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip.eval()
    vis_proj = clip.visual_projection  # Linear(1024 → 768)

    results = {L: [] for L in layers}

    with torch.no_grad():
        for img in tqdm(images, desc="  Extracting"):
            inputs = proc(images=img, return_tensors="pt").to(device)
            out = clip.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
            )
            # hidden_states[0] = embedding, [1] = layer 1, ..., [24] = layer 24
            for L in layers:
                hs = out.hidden_states[L]  # [1, 257, 1024]
                # Project to 768 using CLIP's visual_projection
                projected = vis_proj(hs)  # [1, 257, 768]
                results[L].append(projected.cpu())

    del clip
    torch.cuda.empty_cache()

    return {L: torch.cat(results[L], dim=0) for L in layers}
    # {12: [N, 257, 768], 24: [N, 257, 768]}


def build_embeds(hidden_states, tokenizer, text_encoder, device, mode="cls", n_patch=8):
    """
    Format projected hidden states as [1, 77, 768] for UNet.
    Uses text encoder empty-string output as baseline (proper norm ~28 for BOS/EOS/PAD).

    mode:
      cls:    use only CLS token (position 0), repeated n_tokens times
      patch:  CLS + top-N patch tokens by norm
      mean:   mean of all 257 tokens, repeated
    """
    avg = hidden_states.mean(dim=0)  # [257, 768]
    dtype = next(text_encoder.parameters()).dtype

    # Proper baseline: empty string through text encoder (norm ~28 everywhere)
    with torch.no_grad():
        empty_ids = tokenizer("", padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt").input_ids.to(device)
        baseline = text_encoder(empty_ids)[0]  # [1, 77, 768]

    result = baseline.clone()

    if mode == "cls":
        cls_tok = avg[0].to(device=device, dtype=dtype)
        for i in range(1, 5):
            result[0, i] = cls_tok

    elif mode == "patch":
        cls_tok = avg[0]
        patches = avg[1:]
        norms = patches.norm(dim=-1)
        top_idx = norms.topk(min(n_patch, 75)).indices
        selected = torch.cat([cls_tok.unsqueeze(0), patches[top_idx]], dim=0)
        K = min(selected.shape[0], 75)
        for i in range(K):
            result[0, 1 + i] = selected[i].to(device=device, dtype=dtype)

    elif mode == "mean":
        mean_tok = avg.mean(dim=0).to(device=device, dtype=dtype)
        for i in range(1, 5):
            result[0, i] = mean_tok

    return result  # [1, 77, 768]


def prepare(concept):
    device = torch.device("cuda")
    out_file = EXEMPLAR_DIR / "v26" / f"{concept}.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        print(f"SKIP: {out_file}")
        return

    print(f"\n{'='*60}")
    print(f"v26: {concept}")
    print(f"{'='*60}")

    # Find images
    if concept == "nudity":
        img_dir = EXEMPLAR_DIR / "sd14" / "exemplar_images"
        t_prefix, a_prefix = "nudity_", "clothed_"
        extra_dir = EXEMPLAR_DIR / "sd14" / "exemplar_images_full"
    else:
        img_dir = EXEMPLAR_DIR / "concepts" / concept / "images"
        t_prefix, a_prefix = "target_", "anchor_"
        extra_dir = None

    target_imgs = load_images(img_dir, t_prefix)
    if extra_dir and extra_dir.exists():
        target_imgs += load_images(extra_dir, "full_nudity_")
    anchor_imgs = load_images(img_dir, a_prefix)
    print(f"  Target: {len(target_imgs)}, Anchor: {len(anchor_imgs)}")

    # Extract at layer 12 and 24
    print("  Target hidden states...")
    tgt_hs = extract_hidden_states(target_imgs, device, layers=[12, 24])
    print("  Anchor hidden states...")
    anc_hs = extract_hidden_states(anchor_imgs, device, layers=[12, 24])

    # Load text encoder for formatting
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None).to(device)

    save = {}
    for layer in [12, 24]:
        for mode in ["cls", "patch", "mean"]:
            key_t = f"target_L{layer}_{mode}"
            key_a = f"anchor_L{layer}_{mode}"
            t = build_embeds(tgt_hs[layer], pipe.tokenizer, pipe.text_encoder, device, mode=mode)
            a = build_embeds(anc_hs[layer], pipe.tokenizer, pipe.text_encoder, device, mode=mode)
            save[key_t] = t.cpu().half()
            save[key_a] = a.cpu().half()
            print(f"    L{layer}_{mode}: target norm={t.float().norm(dim=-1).mean():.1f}, "
                  f"anchor norm={a.float().norm(dim=-1).mean():.1f}")

    save["config"] = {
        "concept": concept,
        "layers": [12, 24],
        "modes": ["cls", "patch", "mean"],
        "method": "B_hidden_projection",
    }
    torch.save(save, out_file)
    print(f"  Saved: {out_file}")
    del pipe; torch.cuda.empty_cache()


def main():
    p = ArgumentParser()
    p.add_argument("--concept", default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    concepts = ["nudity", "violence", "harassment", "hate", "shocking", "illegal_activity", "self-harm"]
    if args.all:
        for c in concepts:
            prepare(c)
    elif args.concept:
        prepare(args.concept)


if __name__ == "__main__":
    main()
