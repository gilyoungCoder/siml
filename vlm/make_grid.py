#!/usr/bin/env python
"""
이미지 그리드 생성 스크립트

여러 실험 결과 이미지들을 그리드 형태로 시각화합니다.

Usage:
    # 단일 폴더에서 16개 이미지 샘플링하여 4x4 그리드 생성
    python make_grid.py /path/to/images --output grid.png

    # 여러 폴더 비교 (각 폴더에서 4개씩 뽑아서 비교)
    python make_grid.py /path/to/folder1 /path/to/folder2 --compare --output comparison.png

    # Best configs 자동 비교 (analyze_gpt_results.py 결과 기반)
    python make_grid.py --best-comparison --concept nudity --output best_nudity.png
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL not installed. Run: pip install Pillow")
    sys.exit(1)


def get_images_from_folder(folder: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.webp')) -> List[str]:
    """Get all image paths from a folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    images = []
    for ext in extensions:
        images.extend(folder_path.glob(f'*{ext}'))
        images.extend(folder_path.glob(f'*{ext.upper()}'))

    return sorted([str(p) for p in images])


def sample_images(images: List[str], n: int, seed: int = 42) -> List[str]:
    """Sample n images from the list."""
    random.seed(seed)
    if len(images) <= n:
        return images
    return random.sample(images, n)


def create_grid(
    image_paths: List[str],
    grid_size: Tuple[int, int] = (4, 4),
    cell_size: Tuple[int, int] = (256, 256),
    padding: int = 4,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    title: Optional[str] = None,
    labels: Optional[List[str]] = None
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        image_paths: List of image file paths
        grid_size: (cols, rows) - number of columns and rows
        cell_size: (width, height) of each cell
        padding: Padding between cells
        bg_color: Background color
        title: Optional title at the top
        labels: Optional labels for each image

    Returns:
        PIL Image object
    """
    cols, rows = grid_size
    cell_w, cell_h = cell_size

    # Calculate total size
    title_height = 60 if title else 0
    label_height = 30 if labels else 0

    total_w = cols * cell_w + (cols + 1) * padding
    total_h = rows * (cell_h + label_height) + (rows + 1) * padding + title_height

    # Create canvas
    canvas = Image.new('RGB', (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Try to load a font - larger sizes for better readability
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_w = bbox[2] - bbox[0]
        draw.text(((total_w - text_w) // 2, 15), title, fill=(0, 0, 0), font=title_font)

    # Place images
    for idx, img_path in enumerate(image_paths):
        if idx >= cols * rows:
            break

        row = idx // cols
        col = idx % cols

        x = padding + col * (cell_w + padding)
        y = title_height + padding + row * (cell_h + label_height + padding)

        try:
            img = Image.open(img_path)
            img = img.convert('RGB')

            # Resize maintaining aspect ratio
            img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

            # Center the image in the cell
            paste_x = x + (cell_w - img.width) // 2
            paste_y = y + (cell_h - img.height) // 2

            canvas.paste(img, (paste_x, paste_y))

            # Draw label if provided
            if labels and idx < len(labels):
                label = labels[idx]
                label_y = y + cell_h + 2
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                draw.text((x + (cell_w - text_w) // 2, label_y), label, fill=(100, 100, 100), font=font)

        except Exception as e:
            # Draw placeholder for failed images
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(200, 200, 200))
            draw.text((x + 10, y + cell_h // 2), f"Error: {str(e)[:20]}", fill=(200, 0, 0), font=font)

    return canvas


def create_comparison_grid(
    folders: List[str],
    folder_names: Optional[List[str]] = None,
    images_per_folder: int = 4,
    cell_size: Tuple[int, int] = (256, 256),
    padding: int = 4,
    seed: int = 42,
    title: Optional[str] = None
) -> Image.Image:
    """
    Create a comparison grid with multiple folders.
    Each row shows images from one folder.

    Args:
        folders: List of folder paths
        folder_names: Names for each folder (for labels)
        images_per_folder: Number of images to sample per folder
        cell_size: Size of each image cell
        padding: Padding between cells
        seed: Random seed for sampling
        title: Optional title

    Returns:
        PIL Image object
    """
    if folder_names is None:
        folder_names = [Path(f).name for f in folders]

    cell_w, cell_h = cell_size
    n_folders = len(folders)
    n_cols = images_per_folder

    # Calculate sizes
    title_height = 50 if title else 0
    label_width = 200  # Left column for folder names

    total_w = label_width + n_cols * cell_w + (n_cols + 1) * padding
    total_h = title_height + n_folders * cell_h + (n_folders + 1) * padding

    # Create canvas
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_w = bbox[2] - bbox[0]
        draw.text(((total_w - text_w) // 2, 15), title, fill=(0, 0, 0), font=title_font)

    # Process each folder
    for row_idx, (folder, name) in enumerate(zip(folders, folder_names)):
        y = title_height + padding + row_idx * (cell_h + padding)

        # Draw folder name (truncate if too long)
        display_name = name[:25] + "..." if len(name) > 28 else name
        draw.text((10, y + cell_h // 2 - 6), display_name, fill=(0, 0, 0), font=font)

        # Get and sample images
        images = get_images_from_folder(folder)
        sampled = sample_images(images, images_per_folder, seed=seed)

        # Place images
        for col_idx, img_path in enumerate(sampled):
            x = label_width + padding + col_idx * (cell_w + padding)

            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

                paste_x = x + (cell_w - img.width) // 2
                paste_y = y + (cell_h - img.height) // 2

                canvas.paste(img, (paste_x, paste_y))
            except Exception as e:
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(200, 200, 200))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Create image grids for visualization")
    parser.add_argument('folders', nargs='*', help='Folder(s) containing images')
    parser.add_argument('--output', '-o', type=str, default='grid.png', help='Output file path')
    parser.add_argument('--grid-size', type=str, default='4x4', help='Grid size (e.g., 4x4, 3x3)')
    parser.add_argument('--cell-size', type=int, default=256, help='Size of each cell in pixels')
    parser.add_argument('--compare', action='store_true', help='Compare multiple folders (one row per folder)')
    parser.add_argument('--images-per-folder', type=int, default=4, help='Images per folder in comparison mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--title', type=str, default=None, help='Title for the grid')
    parser.add_argument('--names', type=str, nargs='*', help='Names for folders in comparison mode')

    args = parser.parse_args()

    if not args.folders:
        print("Error: Please provide at least one folder")
        parser.print_help()
        sys.exit(1)

    # Parse grid size
    try:
        cols, rows = map(int, args.grid_size.split('x'))
    except:
        cols, rows = 4, 4

    cell_size = (args.cell_size, args.cell_size)

    if args.compare and len(args.folders) > 1:
        # Comparison mode
        grid = create_comparison_grid(
            folders=args.folders,
            folder_names=args.names,
            images_per_folder=args.images_per_folder,
            cell_size=cell_size,
            seed=args.seed,
            title=args.title
        )
    else:
        # Single folder mode
        folder = args.folders[0]
        images = get_images_from_folder(folder)
        n_images = cols * rows
        sampled = sample_images(images, n_images, seed=args.seed)

        title = args.title or Path(folder).name

        grid = create_grid(
            image_paths=sampled,
            grid_size=(cols, rows),
            cell_size=cell_size,
            title=title
        )

    # Save
    grid.save(args.output, quality=95)
    print(f"✅ Saved grid to: {args.output}")
    print(f"   Size: {grid.width} x {grid.height} pixels")


if __name__ == '__main__':
    main()
