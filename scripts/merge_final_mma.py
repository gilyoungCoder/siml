#!/usr/bin/env python3
"""
generated와 merged 폴더 합치기
merged의 인덱스는 -1 해서 합침
"""
import os
import shutil
import re

BASE = "/mnt/home/yhgil99/unlearning/outputs/nudity_datasets/mma/safree_regenerated"
GENERATED = os.path.join(BASE, "generated")
MERGED = os.path.join(BASE, "merged")
OUTPUT = os.path.join(BASE, "final")

os.makedirs(OUTPUT, exist_ok=True)

count = 0

# 1. generated 폴더 (그대로 복사)
for img in os.listdir(GENERATED):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        shutil.copy2(os.path.join(GENERATED, img), os.path.join(OUTPUT, img))
        count += 1
print(f"Copied {count} from generated")

# 2. merged 폴더 (인덱스 -1)
merged_count = 0
for img in os.listdir(MERGED):
    if not img.endswith(('.png', '.jpg', '.jpeg')):
        continue

    match = re.match(r'^(\d+)_(\d+)_(.+)$', img)
    if match:
        old_idx = int(match.group(1))
        sample_num = match.group(2)
        rest = match.group(3)

        new_idx = old_idx - 1  # 인덱스 -1
        new_name = f"{new_idx:05d}_{sample_num}_{rest}"
    else:
        new_name = img

    src = os.path.join(MERGED, img)
    dst = os.path.join(OUTPUT, new_name)

    # 중복 체크
    if os.path.exists(dst):
        print(f"Skip duplicate: {new_name}")
        continue

    shutil.copy2(src, dst)
    merged_count += 1

print(f"Copied {merged_count} from merged (index -1)")
print(f"\nTotal: {count + merged_count} images in {OUTPUT}")
