#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate single image with 3-class DDPM-noise-injected classifier"
    )
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="Stable Diffusion 체크포인트 폴더 (vae/, scheduler/ 포함)")
    p.add_argument("--classifier_ckpt", type=str, required=True,
                   help="훈련된 분류기 .pth 파일 경로")
    p.add_argument("--image", type=str, required=True,
                   help="분류할 단일 이미지 파일 경로")
    p.add_argument("--batch_size", type=int, default=1,
                   help="배치 크기 (기본 1)")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) VAE & Scheduler 로드
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # 2) 분류기 로드
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    ).to(device)
    classifier.eval()

    # 3) 이미지 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    with open(args.image, "rb") as img_file:
        img = Image.open(img_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    # 4) VAE 인코딩 → latent, 노이즈 주입
    with torch.no_grad():
        lat = vae.encode(img_tensor).latent_dist.sample() * 0.18215  # [1,4,64,64]
        # 원하는 timestep을 직접 지정하거나, 랜덤 샘플링해도 됩니다.
        timestep = torch.tensor([0], device=device, dtype=torch.long)
        noise = torch.randn_like(lat)
        alpha = scheduler.alphas_cumprod[timestep].view(1, *([1]*(lat.ndim-1))).to(device)
        noisy_lat = torch.sqrt(alpha) * lat + torch.sqrt(1 - alpha) * noise

        # 5) 분류기 추론
        logits = classifier(noisy_lat, timestep)  # [1,3]
        probs  = F.softmax(logits, dim=-1)
        pred   = logits.argmax(dim=-1).item()

    # 6) 결과 출력
    class_names = ["benign(사람 없음)", "person(사람 있음)", "nude(사람 누드)"]
    print(f"Predicted class index: {pred}")
    print(f"Class name         : {class_names[pred]}")
    print(f"Confidence scores  : {probs.cpu().numpy().tolist()}")

if __name__ == "__main__":
    main()
