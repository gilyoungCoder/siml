#!/usr/bin/env python3
import os, glob
from PIL import Image, ImageOps

# === 설정 ===
FOLDER = "/mnt/home/yhgil99/unlearning/SAFREE/results/safree_txt"  # 이미지 폴더
OUTFILE = os.path.join(FOLDER, "grid_0-8.png")
ROWS, COLS = 3, 3

def main():
    files = sorted(glob.glob(os.path.join(FOLDER, "*.png")))
    if len(files) < ROWS * COLS:
        raise RuntimeError(f"이미지가 {ROWS*COLS}장 미만입니다. 찾은 개수: {len(files)}")
    files = files[:ROWS * COLS]  # 0~8만 사용

    # 이미지 로드
    imgs = [Image.open(p).convert("RGB") for p in files]
    # 타일 크기(첫 이미지 기준)
    tile_w, tile_h = imgs[0].size

    # 모든 이미지를 동일 크기로 맞춤(필요 시 중앙 크롭/패딩)
    # 같은 해상도면 그대로 유지됨
    imgs = [ImageOps.fit(im, (tile_w, tile_h), method=Image.LANCZOS) for im in imgs]

    # 캔버스 생성
    grid = Image.new("RGB", (COLS * tile_w, ROWS * tile_h), color=(0, 0, 0))

    # 배치
    for i, im in enumerate(imgs):
        r, c = divmod(i, COLS)
        grid.paste(im, (c * tile_w, r * tile_h))

    grid.save(OUTFILE)
    print(f"[SAVE] {OUTFILE}")

if __name__ == "__main__":
    main()
