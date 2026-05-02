#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v25 Embedding Prep: CLIP image CLS × scale (simplest approach)

Takes existing CLIP CLS features [N, 768, norm=1], mean-pools,
scales to target norm, formats as [1, 77, 768] for UNet.

Usage:
  python prepare_clip_v25.py --concept nudity --scale 35
  python prepare_clip_v25.py --all
"""

import torch, torch.nn.functional as F
from pathlib import Path
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline

REPO = Path(__file__).parent.parent
EXEMPLAR_DIR = REPO / "CAS_SpatialCFG" / "exemplars"


def build_scaled_embeds(cls_features, scale, tokenizer, text_encoder, device, n_tokens=4):
    """CLS mean → scale → insert into text encoder baseline output."""
    avg = F.normalize(cls_features.mean(dim=0, keepdim=True), dim=-1)  # [1, 768]
    scaled = avg * scale  # [1, 768, norm=scale]

    # Get proper baseline: encode empty string → all tokens have norm ~28
    dtype = next(text_encoder.parameters()).dtype
    with torch.no_grad():
        empty_ids = tokenizer("", padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt").input_ids.to(device)
        baseline = text_encoder(empty_ids)[0]  # [1, 77, 768] with norm ~28 everywhere

    # Replace token positions 1..n_tokens with scaled concept
    result = baseline.clone()
    concept = scaled.to(device=device, dtype=dtype)
    for i in range(1, 1 + n_tokens):
        result[0, i] = concept[0]

    return result  # [1, 77, 768] — BOS/EOS/PAD all norm ~28, concept norm=scale


def prepare(concept, scales=[25, 30, 35, 40]):
    device = torch.device("cuda")
    out_file = EXEMPLAR_DIR / "v25" / f"{concept}.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        print(f"SKIP: {out_file}")
        return

    # Load existing CLIP CLS features
    if concept == "nudity":
        data = torch.load(EXEMPLAR_DIR / "sd14/clip_exemplar_full_nudity.pt", map_location="cpu")
        tgt_cls = data["target_clip_features"].float()  # [32, 768]
        anc_cls = data["anchor_clip_features"].float()   # [16, 768]
    else:
        data = torch.load(EXEMPLAR_DIR / f"concepts/{concept}/clip_exemplar_projected.pt",
                          map_location="cpu")
        tgt_cls = data["target_clip_features"].float()
        anc_cls = data["anchor_clip_features"].float()

    print(f"  {concept}: target {tgt_cls.shape}, anchor {anc_cls.shape}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None).to(device)

    save = {"target_cls": tgt_cls.half(), "anchor_cls": anc_cls.half()}

    for s in scales:
        tgt = build_scaled_embeds(tgt_cls, s, pipe.tokenizer, pipe.text_encoder, device)
        anc = build_scaled_embeds(anc_cls, s, pipe.tokenizer, pipe.text_encoder, device)
        save[f"target_scale{s}"] = tgt.cpu().half()
        save[f"anchor_scale{s}"] = anc.cpu().half()
        print(f"    scale={s}: norm={tgt.float().norm(dim=-1).mean():.1f}")

    save["config"] = {"concept": concept, "scales": scales, "method": "C_cls_scale"}
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
