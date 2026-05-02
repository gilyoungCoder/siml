#!/usr/bin/env python
"""Generate comparison panels for meeting_20260327.html"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUT = Path("meeting_pack/outputs")
OUT.mkdir(parents=True, exist_ok=True)

BASE = Path("outputs")

# ============================================================
# 1. Side-by-side: Baseline vs v4 vs v13 (same prompt, same seed)
# ============================================================
versions = {
    "SD Baseline": "v3/baseline",
    "v4 (Spatial CAS)": "v4/ainp_a15",
    "v6 (CrossAttn)": "v6/v6_crossattn_ts20_as15",
    "v13 (CLIP Probe)": "v13/ringabell_clip_hybproj_ss10_st03",
}

# Pick several diverse prompts
prompt_indices = [0, 2, 5, 10, 15]  # different scenes
sample_idx = 0  # seed 0

def try_font(size=16):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

font_label = try_font(18)
font_title = try_font(14)

def find_image(directory, pidx, sidx):
    """Find image matching pattern XXXX_YY_*.png"""
    d = BASE / directory
    if not d.exists():
        return None
    prefix = f"{pidx:04d}_{sidx:02d}_"
    for f in sorted(d.iterdir()):
        if f.name.startswith(prefix) and f.suffix == ".png":
            return f
    return None

def get_prompt_text(pidx):
    """Extract prompt from filename."""
    img = find_image("v3/baseline", pidx, 0)
    if img:
        name = img.stem
        # Remove XXXX_YY_ prefix
        text = name[8:].replace("_", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        return text
    return f"Prompt #{pidx}"

# Generate comparison panels
for pidx in prompt_indices:
    images = []
    labels = []
    for label, vdir in versions.items():
        img_path = find_image(vdir, pidx, sample_idx)
        if img_path and img_path.exists():
            img = Image.open(img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
            images.append(img)
            labels.append(label)

    if len(images) < 2:
        continue

    # Create panel
    n = len(images)
    pad = 4
    label_h = 28
    title_h = 24
    W = n * 256 + (n - 1) * pad
    H = 256 + label_h + title_h

    panel = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    # Title
    prompt_text = get_prompt_text(pidx)
    draw.text((8, 2), f'"{prompt_text}"', fill=(100, 100, 100), font=font_title)

    for i, (img, lab) in enumerate(zip(images, labels)):
        x = i * (256 + pad)
        panel.paste(img, (x, title_h))
        # Label
        tw = draw.textlength(lab, font=font_label) if hasattr(draw, 'textlength') else len(lab) * 9
        lx = x + (256 - tw) // 2
        draw.text((lx, title_h + 256 + 2), lab, fill=(30, 30, 30), font=font_label)

    panel.save(OUT / f"compare_{pidx:04d}.png", quality=95)
    print(f"Saved compare_{pidx:04d}.png ({len(images)} versions)")

# ============================================================
# 2. Use existing GradCAM visualizations for mask comparison
# ============================================================
# Copy existing mask vis for presentation
import shutil
for name in ["gradcam_top25_clean.png", "overlay_soft_mask.png", "panel.png"]:
    for scene in ["naked_beach_noguide", "nude_park_noguide"]:
        src = BASE / "mask_vis" / scene / name
        if src.exists():
            dst = OUT / f"mask_{scene}_{name}"
            shutil.copy2(src, dst)
            print(f"Copied {dst.name}")

# ============================================================
# 3. Create a summary panel of v4 spatial CAS mask
# ============================================================
mask_vis_dir = BASE / "mask_vis" / "naked_beach_noguide"
if mask_vis_dir.exists():
    parts = []
    for name in ["final_image.png", "spatial_cas_map_last.png", "soft_mask_last.png", "overlay_soft_mask.png"]:
        p = mask_vis_dir / name
        if p.exists():
            parts.append(Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS))

    if parts:
        labels_mask = ["Generated Image", "Spatial CAS Map", "Soft Mask", "Mask Overlay"]
        n = len(parts)
        W = n * 256 + (n - 1) * 4
        H = 256 + 28
        panel = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (img, lab) in enumerate(zip(parts, labels_mask)):
            x = i * 260
            panel.paste(img, (x, 0))
            tw = draw.textlength(lab, font=font_label) if hasattr(draw, 'textlength') else len(lab) * 9
            draw.text((x + (256 - tw) // 2, 258), lab, fill=(30, 30, 30), font=font_label)
        panel.save(OUT / "v4_mask_pipeline.png", quality=95)
        print("Saved v4_mask_pipeline.png")

print("\nDone! All panels saved to", OUT)
