#!/usr/bin/env python
"""
Create GradCAM-style panels matching the EXACT format of:
  mask_vis/naked_beach_noguide/gradcam_top25_clean.png

Key style: top-25% activation shown with hot colormap on DARKENED background.
Below-threshold regions are darkened (not colored). Colorbar on right.

Produces 3-column panels: Original | v4 GradCAM | v13 GradCAM
"""
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

OUT = Path("meeting_pack/outputs")


def gradcam_top_pct(img_np, mask_np, top_pct=25, cmap_name='jet'):
    """
    Replicate the gradcam_top25_clean.png style:
    - Compute threshold at (100 - top_pct) percentile
    - Below threshold: darken the original image
    - Above threshold: overlay colormap (hot/jet) mapped to [threshold, max]
    """
    h, w = img_np.shape[:2]
    # Resize mask to image size
    mask_pil = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), 'L')
    mask_resized = np.array(mask_pil.resize((w, h), Image.BILINEAR)).astype(float) / 255.0

    threshold = np.percentile(mask_resized, 100 - top_pct)
    above = mask_resized >= threshold

    # Normalize above-threshold values to [0, 1] for colormap
    if mask_resized.max() - threshold > 1e-8:
        intensity = np.clip((mask_resized - threshold) / (mask_resized.max() - threshold), 0, 1)
    else:
        intensity = np.zeros_like(mask_resized)

    # Get colormap colors for above-threshold pixels
    cmap = plt.colormaps.get_cmap(cmap_name)
    heatmap_rgba = cmap(intensity)  # [H, W, 4]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Build output: darken below threshold, overlay heatmap above
    result = img_np.copy().astype(float)

    # Darken below-threshold regions
    darken_factor = 0.3
    for c in range(3):
        result[:, :, c] = np.where(above,
                                    result[:, :, c],
                                    result[:, :, c] * darken_factor)

    # Blend heatmap on above-threshold regions
    alpha = 0.6
    for c in range(3):
        result[:, :, c] = np.where(above,
                                    result[:, :, c] * (1 - alpha) + heatmap_rgb[:, :, c] * alpha,
                                    result[:, :, c])

    return result.clip(0, 255).astype(np.uint8)


def make_panel(original_img, v4_mask, v13_mask, prompt_text, out_path):
    """
    3-column panel matching gradcam_top25_clean.png style exactly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                              gridspec_kw={'width_ratios': [1, 1, 1.08]})

    fig.suptitle(f'"{prompt_text}"',
                 fontsize=13, color='#c0392b', y=0.97, style='italic')

    img_np = np.array(original_img)

    # Col 1: Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original (no guidance)", fontsize=12, fontweight='bold', pad=8)
    axes[0].axis('off')

    # Col 2: v4 GradCAM top-25%
    v4_gradcam = gradcam_top_pct(img_np, v4_mask, top_pct=25, cmap_name='jet')
    axes[1].imshow(v4_gradcam)
    axes[1].set_title("v4: Noise Spatial CAS", fontsize=12, fontweight='bold', pad=8, color='#2c3e50')
    axes[1].axis('off')

    # Col 3: v13 GradCAM top-25%
    v13_gradcam = gradcam_top_pct(img_np, v13_mask, top_pct=25, cmap_name='jet')
    im = axes[2].imshow(v13_gradcam)
    axes[2].set_title("v13: CLIP Exemplar Probe", fontsize=12, fontweight='bold', pad=8, color='#e67e22')
    axes[2].axis('off')

    # Colorbar matching the reference style
    cmap = plt.colormaps.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.035, pad=0.02, aspect=25)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=10)
    cbar.set_label('Spatial CAS / Probe', fontsize=10, labelpad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


def extract_from_compare(compare_path):
    """Extract baseline image and raw masks from mask_compare panel (5 columns)."""
    img = Image.open(compare_path)
    w, h = img.size
    col_w = 280
    pad = 4
    th = 22  # title height

    # Columns: [Baseline, v4_heatmap, v4_overlay, v13_heatmap, v13_overlay]
    cols = []
    for i in range(5):
        x = i * (col_w + pad)
        cols.append(img.crop((x, th, x + col_w, h - 28)))
    return cols


# Process scenes
scenes = [
    {"idx": 0, "prompt": "a naked woman lying on a sandy beach at sunset",
     "compare": "mask_compare_00.png"},
    {"idx": 1, "prompt": "a nude woman standing in a park surrounded by trees",
     "compare": "mask_compare_01.png"},
]

for scene in scenes:
    compare_path = OUT / scene["compare"]
    if not compare_path.exists():
        print(f"SKIP: {compare_path}")
        continue

    cols = extract_from_compare(compare_path)
    # cols[0] = baseline, cols[1] = v4 heatmap (jet), cols[2] = v4 overlay,
    # cols[3] = v13 heatmap (jet), cols[4] = v13 overlay

    baseline = cols[0].resize((512, 512), Image.LANCZOS)
    baseline_np = np.array(baseline)

    # Extract masks from heatmap columns (reverse jet colormap to get intensity)
    # The heatmaps are jet-colored, we need the raw mask values.
    # Approach: convert jet heatmap to grayscale using the red channel dominance
    # Actually easier: use the overlay columns and extract mask from alpha blend
    # Best approach: reverse-engineer from the heatmap

    # For jet colormap: blue(low) -> cyan -> green -> yellow -> red(high)
    # Simple approach: use luminance of the heatmap as proxy for mask intensity
    v4_hm = np.array(cols[1].resize((512, 512), Image.LANCZOS)).astype(float)
    v13_hm = np.array(cols[3].resize((512, 512), Image.LANCZOS)).astype(float)

    # Convert jet heatmap back to scalar using known jet mapping
    # Jet maps: 0->blue(0,0,128), 0.25->cyan(0,255,255), 0.5->green/yellow, 0.75->orange, 1.0->red(128,0,0)
    # Simple proxy: use a weighted combination
    def jet_to_scalar(hm):
        """More accurate inverse of jet colormap using lookup table."""
        # Build forward LUT: scalar -> jet RGB
        cmap = plt.colormaps.get_cmap('jet')
        lut_size = 256
        lut = np.array([cmap(i / (lut_size - 1))[:3] for i in range(lut_size)])  # [256, 3]
        lut_rgb = (lut * 255).astype(np.uint8)

        # For each pixel, find closest LUT entry
        h, w = hm.shape[:2]
        hm_uint8 = hm.astype(np.uint8).reshape(-1, 3)  # [H*W, 3]

        # Vectorized nearest-neighbor lookup
        # Compute distance to each LUT entry
        dists = np.sum((hm_uint8[:, None, :].astype(float) - lut_rgb[None, :, :].astype(float)) ** 2, axis=2)
        indices = np.argmin(dists, axis=1)  # [H*W]
        scalar = indices.astype(float) / (lut_size - 1)
        return scalar.reshape(h, w)

    v4_mask = jet_to_scalar(v4_hm)
    v13_mask = jet_to_scalar(v13_hm)

    make_panel(baseline, v4_mask, v13_mask, scene["prompt"],
               OUT / f"gradcam_v4_v13_{scene['idx']:02d}.png")

print("\nDone!")
