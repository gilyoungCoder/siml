from PIL import Image
import os

# 이미지 경로 설정
base_dir = "./"
image_files = [f"{i:06}.png" for i in range(9)]  # 000000.png ~ 000008.png

# 이미지 불러오기
images = [Image.open(os.path.join(base_dir, file)) for file in image_files]

# 첫 이미지 기준으로 크기 설정
w, h = images[0].size
grid_image = Image.new('RGB', (3 * w, 3 * h))  # 3x3 격자

# 이미지 배치
for idx, img in enumerate(images):
    row, col = divmod(idx, 3)
    grid_image.paste(img, (col * w, row * h))

# 저장
output_path = os.path.join(base_dir, "grid_3x3.png")
grid_image.save(output_path)
print(f"Saved to {output_path}")
