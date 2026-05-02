#!/usr/bin/env python
"""
Combine individual method grids into one concept-level comparison grid.
Arranges in 2xN grid layout for better viewing.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import math

def get_method_order(concept):
    """Define the order of methods for each concept."""
    if concept == 'nudity':
        return [
            'SD1.4',
            'SAFREE',
            'SAFREE+Ours_4class',
            'Ours_4class_noskip',
            'Ours_4class_skip',
            'Ours_6class_noskip',
            'Ours_6class_skip',
        ]
    elif concept == 'violence':
        return [
            'SD1.4',
            'SAFREE',
            'SAFREE+Ours_9class',
            'Ours_9class_skip',
            'Ours_9class_noskip',
            'Ours_13class_skip',
            'Ours_13class_noskip',
        ]
    else:
        return [
            'SD1.4',
            'SAFREE',
            'SAFREE+Ours_9class',
            'Ours_9class_skip',
            'Ours_9class_noskip',
        ]

def get_display_name(method):
    """Convert filename method to display name."""
    mapping = {
        'SD1.4': 'SD 1.4',
        'SAFREE': 'SAFREE',
        'SAFREE+Ours_4class': 'SAFREE+Ours (4class)',
        'SAFREE+Ours_9class': 'SAFREE+Ours (9class)',
        'Ours_4class_noskip': 'Ours 4class (noskip)',
        'Ours_4class_skip': 'Ours 4class (skip)',
        'Ours_6class_noskip': 'Ours 6class (noskip)',
        'Ours_6class_skip': 'Ours 6class (skip)',
        'Ours_9class_skip': 'Ours 9class (skip)',
        'Ours_9class_noskip': 'Ours 9class (noskip)',
        'Ours_13class_skip': 'Ours 13class (skip)',
        'Ours_13class_noskip': 'Ours 13class (noskip)',
    }
    return mapping.get(method, method)

def combine_concept_grids(grids_dir, output_dir=None, cols=2):
    """Combine grids by concept in NxM grid layout."""
    grids_path = Path(grids_dir)
    output_path = Path(output_dir) if output_dir else grids_path

    # Group images by concept
    concept_images = defaultdict(dict)

    for img_file in grids_path.glob('*.png'):
        if img_file.name.endswith('_combined.png'):
            continue

        # Parse filename: concept_method.png
        name = img_file.stem  # e.g., nudity_SD1.4
        parts = name.split('_', 1)
        if len(parts) == 2:
            concept, method = parts
            concept_images[concept][method] = img_file

    # Try to load fonts - larger sizes
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        label_font = ImageFont.load_default()
        title_font = label_font

    # Process each concept
    for concept, methods in concept_images.items():
        print(f"\nProcessing {concept}...")

        # Get ordered methods
        method_order = get_method_order(concept)
        ordered_methods = [(m, methods[m]) for m in method_order if m in methods]

        if not ordered_methods:
            print(f"  No images found for {concept}")
            continue

        # Load first image to get dimensions
        sample_img = Image.open(ordered_methods[0][1])
        img_width, img_height = sample_img.size

        # Calculate grid layout
        n_methods = len(ordered_methods)
        n_cols = cols
        n_rows = math.ceil(n_methods / n_cols)

        # Layout params
        label_height = 50  # Height for method label above each grid
        title_height = 70  # Height for concept title
        padding = 15

        # Calculate canvas size
        cell_width = img_width + padding
        cell_height = img_height + label_height + padding

        total_width = n_cols * cell_width + padding
        total_height = title_height + n_rows * cell_height + padding

        canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw title
        title_text = concept.upper()
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        text_w = bbox[2] - bbox[0]
        draw.text(((total_width - text_w) // 2, 15), title_text, fill=(0, 0, 0), font=title_font)

        # Place each method grid in 2xN layout
        for idx, (method, img_path) in enumerate(ordered_methods):
            row = idx // n_cols
            col = idx % n_cols

            x = padding + col * cell_width
            y = title_height + row * cell_height

            # Draw method label
            display_name = get_display_name(method)
            bbox = draw.textbbox((0, 0), display_name, font=label_font)
            text_w = bbox[2] - bbox[0]
            label_x = x + (img_width - text_w) // 2
            draw.text((label_x, y), display_name, fill=(30, 30, 30), font=label_font)

            # Paste grid image
            img = Image.open(img_path)
            canvas.paste(img, (x, y + label_height))

            print(f"  Added: {display_name} at ({col}, {row})")

        # Save combined grid
        output_file = output_path / f"{concept}_combined.png"
        canvas.save(output_file, quality=95)
        print(f"  Saved: {output_file}")
        print(f"  Size: {canvas.width} x {canvas.height}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combine method grids by concept")
    parser.add_argument('--grids-dir', default='/mnt/home/yhgil99/unlearning/vlm/grids',
                        help='Directory containing grid images')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as grids-dir)')
    parser.add_argument('--cols', type=int, default=2,
                        help='Number of columns in combined grid (default: 2)')

    args = parser.parse_args()
    combine_concept_grids(args.grids_dir, args.output_dir, args.cols)

if __name__ == '__main__':
    main()
