#!/usr/bin/env python3
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
from diffusers import DDPMScheduler

# 1) 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) custom collate_fn: PIL.Image 리스트와 정수 라벨 텐서를 반환
def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)

# 3) 데이터셋 로드 (원본 32×32 PIL.Image 그대로)
root_dir = '/mnt/home/yhgil99/dataset/clip/cifar-10-batches-py/cifar10_test_subset'
dataset  = ImageFolder(root=root_dir, transform=lambda x: x)
loader   = DataLoader(
    dataset, batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=collate_fn
)

# 4) CLIP 모델 & 프로세서 초기화
PRETRAINED = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(PRETRAINED)
model     = CLIPModel.from_pretrained(PRETRAINED).to(device).eval()

# 5) 텍스트 임베딩 계산 (한 번만)
class_names = dataset.classes
prompts     = [f"a photo of a {c}." for c in class_names]
with torch.no_grad():
    text_inputs     = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_embeddings = model.get_text_features(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
    )
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# 6) Diffusion forward noise scheduler 설정 (T timesteps)
scheduler = DDPMScheduler(
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=0.02,
    num_train_timesteps=50,
)
alphas_cumprod = scheduler.alphas_cumprod.to(device)  # length == T

# 7) 5개 구간의 대표 timestep 계산
T = scheduler.num_train_timesteps
# 경계 6개: 0, T/5, 2T/5, ..., T
bounds = torch.linspace(0, T, steps=6, dtype=torch.long).clamp(max=T-1)
# 각 구간의 중간값
time_steps = [int((bounds[i] + bounds[i+1]) // 2) for i in range(5)]

# 8) 구간별 평가
results = {}
print("=== Evaluating CLIP zero-shot under 5 diffusion noise levels ===")
for t in time_steps:
    correct = 0
    total   = 0
    α_bar   = alphas_cumprod[t]                  # ᾱ_t
    sqrt_α  = α_bar.sqrt()                       # √ᾱ_t
    sqrt_1m = (1 - α_bar).sqrt()                 # √(1−ᾱ_t)

    pbar = tqdm(loader, desc=f" t={t:2d} ", leave=False)
    for images, labels in pbar:
        # 8.1) 이미지 전처리: 자동 리사이즈(224×224) & 정규화
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        x0     = inputs.pixel_values  # shape: [B,3,224,224]

        # 8.2) forward diffusion 노이즈 주입
        noise = torch.randn_like(x0)
        xt    = sqrt_α * x0 + sqrt_1m * noise

        # 8.3) CLIP 임베딩 & 분류
        with torch.no_grad():
            emb = model.get_image_features(pixel_values=xt)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = emb @ text_embeddings.T
            preds  = logits.argmax(dim=-1)

        correct += (preds.cpu() == labels).sum().item()
        total   += labels.size(0)

    acc = 100 * correct / total
    results[t] = acc
    print(f">>> Representative t = {t:2d}: Accuracy = {acc:5.2f}%")

# 9) 결과 요약 출력
print("\n=== Summary: CLIP accuracy vs. diffusion noise segments ===")
for t in time_steps:
    print(f" t = {t:2d} → {results[t]:5.2f}%")
