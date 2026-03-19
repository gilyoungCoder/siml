#!/usr/bin/env python3
"""
MMA safree_regenerated 하위 폴더들의 이미지를 하나로 합치고
파일명 인덱스를 실제 인덱스에 맞게 조정
"""
import os
import shutil
import re
from pathlib import Path

BASE = "/mnt/home/yhgil99/unlearning/outputs/nudity_datasets/mma/safree_regenerated"
OUTPUT = os.path.join(BASE, "merged")

os.makedirs(OUTPUT, exist_ok=True)

# 하위 폴더들 찾기 (300_387, 388_475, ... 형식)
subdirs = []
for d in os.listdir(BASE):
    if re.match(r'^\d+_\d+$', d):
        subdirs.append(d)

subdirs.sort(key=lambda x: int(x.split('_')[0]))
print(f"Found {len(subdirs)} subdirs: {subdirs}")

count = 0
for subdir in subdirs:
    start_idx = int(subdir.split('_')[0])

    # generated 폴더 안에 있을 수도 있음
    img_dir = os.path.join(BASE, subdir, "generated")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(BASE, subdir)

    if not os.path.exists(img_dir):
        print(f"Skip: {subdir} (not found)")
        continue

    # 이미지 파일들 찾기
    images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    for img in images:
        # 파일명에서 기존 인덱스 추출: 00000_00_xxx.png
        match = re.match(r'^(\d+)_(\d+)_(.+)$', img)
        if match:
            old_idx = int(match.group(1))
            sample_num = match.group(2)
            rest = match.group(3)

            # 실제 인덱스 계산
            real_idx = start_idx + old_idx
            new_name = f"{real_idx:05d}_{sample_num}_{rest}"
        else:
            # 패턴 안맞으면 그냥 start_idx 기반으로
            real_idx = start_idx + count
            new_name = f"{real_idx:05d}_{img}"

        src = os.path.join(img_dir, img)
        dst = os.path.join(OUTPUT, new_name)
        shutil.copy2(src, dst)
        count += 1

print(f"\nDone! Merged {count} images to {OUTPUT}")
