#!/usr/bin/env python
"""
FID + CLIP Score Evaluation.

Computes FID between baseline and method image directories, and CLIP score
for text-image alignment.

Usage:
    python -m evaluation.eval_fid_clip <baseline_dir> <method_dir> <prompts_txt>
"""

import sys
import os
import glob
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore
from pytorch_fid.fid_score import calculate_fid_given_paths


def load_prompts(txt_path, nsamples=1):
    """Load prompts, repeat nsamples times to match image count."""
    with open(txt_path) as f:
        prompts = [l.strip() for l in f if l.strip()]
    return [p for p in prompts for _ in range(nsamples)]


def get_images_sorted(img_dir):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        imgs += glob.glob(os.path.join(img_dir, ext))
    return sorted(imgs)


def compute_clip_score(img_dir, prompts, device="cuda", batch_size=32):
    """Compute average CLIP score."""
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    imgs = get_images_sorted(img_dir)
    n = min(len(imgs), len(prompts))

    to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    scores = []

    for i in range(0, n, batch_size):
        batch_imgs = imgs[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        tensors = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in batch_imgs]).to(device)
        tensors_uint8 = (tensors * 255).to(torch.uint8)
        score = metric(tensors_uint8, batch_prompts)
        scores.append(score.item())

    return sum(scores) / len(scores) if scores else 0.0


def compute_fid(dir1, dir2, device="cuda", batch_size=64):
    """Compute FID between two directories."""
    return calculate_fid_given_paths(
        [dir1, dir2], batch_size=batch_size, device=device, dims=2048, num_workers=4,
    )


def main():
    if len(sys.argv) < 4:
        print("Usage: eval_fid_clip.py <baseline_dir> <method_dir> <prompts_txt>")
        sys.exit(1)

    baseline_dir, method_dir, prompts_txt = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(prompts_txt)

    print("Computing CLIP scores...")
    clip_base = compute_clip_score(baseline_dir, prompts, device)
    clip_method = compute_clip_score(method_dir, prompts, device)
    print(f"  Baseline CLIP: {clip_base:.4f}")
    print(f"  Method CLIP:   {clip_method:.4f}")

    print("\nComputing FID...")
    fid = compute_fid(baseline_dir, method_dir, device)
    print(f"  FID: {fid:.2f}")

    out = Path(method_dir) / "results_fid_clip.txt"
    with open(out, "w") as f:
        f.write(f"CLIP (Baseline): {clip_base:.4f}\n")
        f.write(f"CLIP (Method):   {clip_method:.4f}\n")
        f.write(f"FID:             {fid:.2f}\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
