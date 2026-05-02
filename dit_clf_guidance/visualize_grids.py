#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create 4x4 grid visualizations of generated images.

Generates:
  1. Baseline grid: 4x4 grid of baseline (no guidance) images
  2. Comparison grid: Same prompts across baseline / s5 / s10 / s20 (4 columns, 4 rows)
"""

import os
from PIL import Image, ImageDraw, ImageFont


def load_prompts(prompt_path):
    """Load prompts from text file."""
    with open(prompt_path) as f:
        return [line.strip() for line in f if line.strip()]


def make_grid(images, ncols=4, cell_size=256, padding=4, bg_color=(255, 255, 255)):
    """Create a grid image from a list of PIL images."""
    nrows = (len(images) + ncols - 1) // ncols
    w = ncols * cell_size + (ncols + 1) * padding
    h = nrows * cell_size + (nrows + 1) * padding
    grid = Image.new("RGB", (w, h), bg_color)

    for i, img in enumerate(images):
        row, col = divmod(i, ncols)
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + padding)
        resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(resized, (x, y))

    return grid


def make_comparison_grid(dirs, labels, indices, cell_size=256, padding=4):
    """
    Create a comparison grid: rows = prompts, columns = guidance scales.

    Args:
        dirs: list of directories (one per column)
        labels: list of column labels
        indices: list of image indices to show (one per row)
    """
    ncols = len(dirs)
    nrows = len(indices)

    # Header height
    header_h = 40
    w = ncols * cell_size + (ncols + 1) * padding
    h = header_h + nrows * cell_size + (nrows + 1) * padding

    grid = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Draw column headers
    for col, label in enumerate(labels):
        x = padding + col * (cell_size + padding) + cell_size // 2
        draw.text((x, 10), label, fill=(0, 0, 0), font=font, anchor="mt")

    # Draw images
    for row, idx in enumerate(indices):
        for col, d in enumerate(dirs):
            fname = f"{idx:06d}.png"
            img_path = os.path.join(d, fname)
            if os.path.exists(img_path):
                img = Image.open(img_path).resize((cell_size, cell_size), Image.LANCZOS)
            else:
                # Placeholder
                img = Image.new("RGB", (cell_size, cell_size), (200, 200, 200))
            x = padding + col * (cell_size + padding)
            y = header_h + padding + row * (cell_size + padding)
            grid.paste(img, (x, y))

    return grid


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "visualizations")
    img_dir = os.path.join(base_dir, "output_img")
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. Country Nude Body — Baseline 4x4 grid
    # ================================================================
    print("=== Country Nude Body Baseline Grid ===")
    country_baseline_dir = os.path.join(img_dir, "country_nude_body_baseline")
    prompts_path = os.path.join(country_baseline_dir, "prompts.txt")
    if os.path.exists(prompts_path):
        prompts = load_prompts(prompts_path)
    else:
        prompts = [f"prompt_{i}" for i in range(50)]

    # Pick 16 images (first 16)
    images = []
    for i in range(min(16, len(prompts))):
        img_path = os.path.join(country_baseline_dir, f"{i:06d}.png")
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
    if images:
        grid = make_grid(images, ncols=4, cell_size=256)
        save_path = os.path.join(output_dir, "country_nude_body_baseline_grid.png")
        grid.save(save_path)
        print(f"  Saved: {save_path} ({len(images)} images)")

    # ================================================================
    # 2. Country Nude Body — Guided Comparison (Baseline vs s5 vs s10 vs s20)
    # ================================================================
    print("=== Country Nude Body Guided Comparison ===")
    dirs = [
        os.path.join(img_dir, "country_nude_body_baseline"),
        os.path.join(img_dir, "country_nude_body_guided_s5"),
        os.path.join(img_dir, "country_nude_body_guided_s10"),
        os.path.join(img_dir, "country_nude_body_guided_s20"),
    ]
    labels = ["Baseline", "Scale=5", "Scale=10", "Scale=20"]
    # Show 8 diverse prompts (indices: Japan, US, Korea, India, Brazil, France, Italy, Germany)
    indices = [2, 0, 11, 4, 9, 6, 7, 3]  # Japan, US, S.Korea, India, Brazil, France, Italy, Germany
    comp = make_comparison_grid(dirs, labels, indices, cell_size=256)
    save_path = os.path.join(output_dir, "country_nude_body_guided_comparison.png")
    comp.save(save_path)
    print(f"  Saved: {save_path} ({len(indices)} rows x {len(labels)} cols)")

    # ================================================================
    # 3. Ring-a-Bell — Baseline 4x4 grid
    # ================================================================
    print("=== Ring-a-Bell Baseline Grid ===")
    ringabell_baseline_dir = os.path.join(img_dir, "ringabell_nudity_baseline")
    images = []
    for i in range(16):
        img_path = os.path.join(ringabell_baseline_dir, f"{i:06d}.png")
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
    if images:
        grid = make_grid(images, ncols=4, cell_size=256)
        save_path = os.path.join(output_dir, "ringabell_baseline_grid.png")
        grid.save(save_path)
        print(f"  Saved: {save_path} ({len(images)} images)")

    # ================================================================
    # 4. Ring-a-Bell — Guided Comparison
    # ================================================================
    print("=== Ring-a-Bell Guided Comparison ===")
    dirs = [
        os.path.join(img_dir, "ringabell_nudity_baseline"),
        os.path.join(img_dir, "ringabell_nudity_guided_s5"),
        os.path.join(img_dir, "ringabell_nudity_guided_s10"),
        os.path.join(img_dir, "ringabell_nudity_guided_s20"),
    ]
    labels = ["Baseline", "Scale=5", "Scale=10", "Scale=20"]
    indices = list(range(8))  # First 8 prompts
    comp = make_comparison_grid(dirs, labels, indices, cell_size=256)
    save_path = os.path.join(output_dir, "ringabell_guided_comparison.png")
    comp.save(save_path)
    print(f"  Saved: {save_path} ({len(indices)} rows x {len(labels)} cols)")

    # ================================================================
    # 5. Country Body (non-nude) — Guided Comparison (should preserve quality)
    # ================================================================
    print("=== Country Body (non-nude) Guided Comparison ===")
    country_body_dirs = [
        os.path.join(img_dir, "country_body_guided_s5"),
        os.path.join(img_dir, "country_body_guided_s10"),
        os.path.join(img_dir, "country_body_guided_s15"),
        os.path.join(img_dir, "country_body_guided_s20"),
    ]
    # Check if these exist
    if all(os.path.isdir(d) for d in country_body_dirs):
        labels = ["Scale=5", "Scale=10", "Scale=15", "Scale=20"]
        indices = [2, 0, 11, 4, 9, 6, 7, 3]  # Same countries
        comp = make_comparison_grid(country_body_dirs, labels, indices, cell_size=256)
        save_path = os.path.join(output_dir, "country_body_guided_comparison.png")
        comp.save(save_path)
        print(f"  Saved: {save_path}")

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
