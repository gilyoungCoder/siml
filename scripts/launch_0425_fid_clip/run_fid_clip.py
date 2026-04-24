#!/usr/bin/env python3
"""FID (pytorch_fid) + CLIP score (ViT-L/14 via clip package).
Computes:
  - FID(baseline vs anchor), FID(baseline vs hybrid)
  - CLIP cosine per (baseline, anchor, hybrid) over COCO 10k prompts
"""
import os, glob, sys, torch
from PIL import Image
import clip
from pytorch_fid.fid_score import calculate_fid_given_paths

ROOT = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0424_coco10k"
PROMPTS_TXT = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/coco_10k.txt"
BASE = f"{ROOT}/baseline_sd14"
ANCHOR = f"{ROOT}/ours_anchor_v2pack"
HYBRID = f"{ROOT}/ours_hybrid_v1pack"
device = "cuda"
dims = 2048
batch_size = 64

prompts = [l.strip() for l in open(PROMPTS_TXT) if l.strip()]
print(f"Loaded {len(prompts)} COCO prompts", flush=True)

def imgs_sorted(d):
    out = []
    for ext in ["*.png","*.jpg","*.jpeg"]:
        out += glob.glob(os.path.join(d, ext))
    return sorted(out)

print("Loading CLIP ViT-L/14...")
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()

def clip_cosine(img_dir, prompts):
    imgs = imgs_sorted(img_dir)
    n = min(len(imgs), len(prompts))
    imgs = imgs[:n]; prompts = prompts[:n]
    total = 0.0; count = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bi = imgs[i:i+batch_size]
            bp = prompts[i:i+batch_size]
            img_t = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in bi]).to(device)
            txt_t = clip.tokenize(bp, truncate=True).to(device)
            img_f = model.encode_image(img_t)
            txt_f = model.encode_text(txt_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sim = (img_f * txt_f).sum(dim=-1)
            total += sim.sum().item()
            count += len(bi)
            if (i//batch_size) % 20 == 0:
                print(f"  CLIP batch {i//batch_size+1}/{(n+batch_size-1)//batch_size} (running avg={total/count:.4f})", flush=True)
    return total / count

results = {}
for label, d in [("baseline", BASE), ("anchor", ANCHOR), ("hybrid", HYBRID)]:
    n = len(imgs_sorted(d))
    print(f"\n=== CLIP score: {label} ({d}: {n} imgs) ===")
    if n == 0: print("  SKIP"); continue
    c = clip_cosine(d, prompts)
    print(f"  CLIP cosine = {c:.4f}")
    results[f"{label}_clip"] = c

# Free CLIP before loading Inception for FID
del model
torch.cuda.empty_cache()

for label, d in [("anchor", ANCHOR), ("hybrid", HYBRID)]:
    print(f"\n=== FID: baseline vs {label} ===")
    try:
        fid = calculate_fid_given_paths([BASE, d], batch_size=batch_size, device=device, dims=dims, num_workers=4)
        print(f"  FID = {fid:.4f}")
        results[f"fid_baseline_vs_{label}"] = fid
    except Exception as e:
        print(f"  FAILED: {e}")

print("\n=== FINAL ===")
for k, v in results.items(): print(f"  {k} = {v:.4f}")
