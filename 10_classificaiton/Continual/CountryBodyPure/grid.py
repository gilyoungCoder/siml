#!/usr/bin/env python3
import os
from PIL import Image

def make_grid(
    root_dir: str,
    scale_folders: list,
    img_names: list,
    output_path: str
):
    """
    root_dir:      "CountryBodyFixed" 폴더 경로
    scale_folders: ["5", "10", "15", "20", "25"] 등 하위 폴더 이름 리스트
    img_names:     ["1.png", "2.png", "3.png"] 같이, 각 폴더에서 붙일 파일 이름 리스트
    output_path:   결과로 저장할 경로 (예: "grid.png")
    """
    # (1) 첫 번째 이미지를 열어 크기 가정. 모든 이미지가 동일한 크기여야 합니다.
    sample_path = os.path.join(root_dir, scale_folders[0], img_names[0])
    with Image.open(sample_path) as im:
        img_w, img_h = im.size

    cols = len(img_names)       # 가로로 붙일 이미지 개수, 여기서는 3
    rows = len(scale_folders)   # 세로로 붙일 스케일 개수, 여기서는 5

    # (2) 전체 그리드 크기 계산
    grid_w = img_w * cols
    grid_h = img_h * rows

    # (3) 빈 캔버스 생성 (RGBA 모드로 하면 투명 배경 가능)
    grid_image = Image.new("RGB", (grid_w, grid_h), color=(255,255,255))

    # (4) 각 셀에 이미지 붙이기
    for row_idx, scale in enumerate(scale_folders):
        folder_path = os.path.join(root_dir, scale)
        for col_idx, img_name in enumerate(img_names):
            img_path = os.path.join(folder_path, img_name)
            if not os.path.isfile(img_path):
                print(f"경고: {img_path} 파일을 찾을 수 없습니다. 스킵합니다.")
                continue

            with Image.open(img_path) as im:
                # 필요하다면 im = im.resize((img_w, img_h))  # 크기를 강제로 통일하려면 이 줄을 사용
                # 붙여넣을 좌표 계산
                x = col_idx * img_w
                y = row_idx * img_h
                grid_image.paste(im, (x, y))

    # (5) 최종 이미지 저장
    grid_image.save(output_path)
    print(f"완료: {output_path} 파일이 생성되었습니다.")

if __name__ == "__main__":
    # 예시: 현재 스크립트와 같은 경로에 "CountryBodyFixed" 폴더가 있다고 가정
    base_folder = "./"  # 실제 경로로 변경하세요
    scales = ["5", "10", "15", "20", "25"]
    names = ["2.png", "3.png", "5.png"]
    out_file = "./CountryBodyFixed_grid25.png"

    make_grid(base_folder, scales, names, out_file)