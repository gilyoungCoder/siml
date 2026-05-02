#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

# Latent→CLIP 투영 네트워크 (훈련된 체크포인트로 로드)
class Latent2Clip(torch.nn.Module):
    def __init__(self, latent_dim, clip_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, clip_dim),
        )
    def forward(self, z):
        proj = self.net(z)
        return proj / proj.norm(dim=-1, keepdim=True)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = '/mnt/home/yhgil99/dataset/clip/cifar-10-batches-py/cifar10_test_subset'

    # 1) 데이터 로더 (PIL Image 그대로)
    ds = ImageFolder(root=root_dir, transform=lambda img: img)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, pin_memory=True,
                        collate_fn=collate_fn)
    class_names = ds.classes  # CIFAR-10 클래스

    # 2) VAE & Latent2Clip & CLIP 초기화
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")\
                       .to(device).eval()
    # 투영 네트워크
    latent2clip = Latent2Clip(vae.config.latent_channels,
                              CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                                       .config.projection_dim).to(device)
    ckpt = torch.load("./checkpoints/latent2clip_epoch27.pt",
                     map_location=device)
    latent2clip.load_state_dict(ckpt["model_state"])
    latent2clip.eval()

    # 3) CLIP 모델 & processor, 텍스트 임베딩
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")\
                    .to(device).eval()

    with torch.no_grad():
        text_inputs = processor(text=[f"a photo of a {c}." for c in class_names],
                                return_tensors="pt", padding=True).to(device)
        text_embeds = clip.get_text_features(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    # 4) Diffusion 스케줄러 정의 (T=50)
    scheduler = DDPMScheduler(beta_schedule="linear",
                              beta_start=0.0001,
                              beta_end=0.02,
                              num_train_timesteps=50)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # 5) 5개 구간 대표 timestep
    T = scheduler.num_train_timesteps
    bounds = torch.linspace(0, T, steps=6, dtype=torch.long).clamp(max=T-1)
    time_steps = [int((bounds[i]+bounds[i+1])//2) for i in range(5)]

    # 6) latent 노이즈 injection → 투영 → 분류 평가
    vae_preproc = transforms.Compose([
        transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    print("=== Evaluating Latent2Clip under noisy latents ===")
    results = {}
    for t in time_steps:
        α_bar = alphas_cumprod[t]
        sqrt_α = α_bar.sqrt()
        sqrt_1m = (1 - α_bar).sqrt()

        correct = 0; total = 0
        pbar = tqdm(loader, desc=f"t={t:2d}", leave=False)
        for images, labels in pbar:
            # (a) VAE latent
            x = torch.stack([vae_preproc(img) for img in images]).to(device)
            with torch.no_grad():
                lat_dist = vae.encode(x).latent_dist
                lat0 = lat_dist.sample() * vae.config.scaling_factor  # (B,C,H,W)

            # (b) latent forward diffusion
            noise = torch.randn_like(lat0)
            lat_t = sqrt_α * lat0 + sqrt_1m * noise

            # (c) projection → CLIP 분류
            with torch.no_grad():
                proj_feat = latent2clip(lat_t)
                logits = proj_feat @ text_embeds.T
                preds = logits.argmax(dim=-1)

            correct += (preds.cpu() == labels).sum().item()
            total += len(labels)

        acc = 100 * correct / total
        results[t] = acc
        print(f">>> t={t:2d}: Latent2Clip ZS acc = {acc:5.2f}%")

    print("\n=== Summary ===")
    for t in time_steps:
        print(f" t = {t:2d} → {results[t]:5.2f}%")

if __name__ == "__main__":
    main()
