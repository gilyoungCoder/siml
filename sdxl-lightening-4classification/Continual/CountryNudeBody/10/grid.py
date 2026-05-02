from PIL import Image
import matplotlib.pyplot as plt
import os

# 이미지 경로 불러오기
image_paths = sorted([
    f for f in os.listdir("./")
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:9]  # 상위 9개 이미지만 사용

# 이미지 로드 및 리사이즈
images = [Image.open(p).resize((256, 256)) for p in image_paths]

# 3x3 그리드 생성
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i])
    ax.axis("off")

plt.tight_layout()
plt.savefig("grid_output.png", bbox_inches='tight')
plt.close()
