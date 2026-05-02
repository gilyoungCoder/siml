#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image

def merge_images_horizontally(image_paths):
    """주어진 이미지 경로 리스트를 가로로 이어붙여 하나의 이미지로 반환."""
    images = [Image.open(p) for p in image_paths]
    # 높이는 가장 큰 이미지 높이로, 너비는 모두 합산
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    merged = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width
    return merged

def main(root_dir: Path):
    # "discriminator+"로 시작하는 모든 하위 폴더 순회
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith('discriminator'):
            continue

        # 1.png, 2.png, 3.png 확인
        img_files = [folder / f"{i}.png" for i in (1, 2, 3)]
        if any(not p.exists() for p in img_files):
            print(f"[SKIP] {folder} 에서 1.png~3.png 중 일부를 찾을 수 없습니다.")
            continue

        # 이미지 합치기
        merged = merge_images_horizontally(img_files)
        out_path = folder / "merged.png"
        merged.save(out_path)
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="discriminator+* 폴더 안의 1.png,2.png,3.png를 가로로 합쳐 merged.png로 저장"
    )
    parser.add_argument(
        "--root", "-r",
        type=Path,
        default=Path("."),
        help="탐색을 시작할 루트 디렉토리 (기본: 현재 폴더)"
    )
    args = parser.parse_args()
    main(args.root)
