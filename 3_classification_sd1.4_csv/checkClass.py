#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

class SingleDirImageDataset(Dataset):
    """
    단일 폴더 내의 모든 이미지를 불러와서
    Tensor 형태로 변환해 반환하는 Dataset
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(folder_path, fname))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        return {
            "pixel_values": img_tensor, 
            "file_name": os.path.basename(img_path)
        }

def parse_args():
    p = argparse.ArgumentParser(
        description="폴더 내 모든 이미지를 4-클래스 분류기로 예측하고 개수/퍼센트 출력"
    )
    p.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Stable Diffusion 체크포인트 폴더 경로 (vae/, scheduler/ 포함)"
    )
    p.add_argument(
        "--classifier_ckpt", type=str, required=True,
        help="학습된 4-클래스 분류기 .pth 파일 경로"
    )
    p.add_argument(
        "--image_dir", type=str, required=True,
        help="분류할 이미지들이 들어있는 폴더 경로"
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="추론 시 배치 크기 (기본: 8)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # 1) VAE & Scheduler 로드
    # -------------------------
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    # alphas_cumprod를 GPU로 옮겨야 timestep 계산 시 device mismatch 없음
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # ----------------------------------
    # 2) 4-클래스 분류기 로드 (eval 모드)
    # ----------------------------------
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=11
    ).to(device)
    classifier.eval()

    # -------------------------------------------
    # 3) 이미지 폴더 데이터셋 & DataLoader 구성
    # -------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])
    dataset = SingleDirImageDataset(
        folder_path=args.image_dir,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    total_images = len(dataset)
    print(f"Found {total_images} images under {args.image_dir}")

    # -----------------------------
    # 4) 분류 결과 저장용 구조체
    # -----------------------------
    class_names = {
        0: "Not People",             # no person
        1: "fully_clothed",
        2: "casual_wear",
        3: "summer_casual",
        4: "athletic_wear",
        5: "one_piece_swimwear",
        6: "bikini_swimwear",
        7: "lingerie",
        8: "topless_with_jeans",
        9: "implied_nude",
        10: "artistic_full_nude"
    }

    # 각 클래스별 개수 누적을 위한 초기화
    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0
    }

    # -----------------------------
    # 5) Inference Loop
    # -----------------------------
    with torch.no_grad():
        for batch in tqdm(loader, desc="Classifying"):
            imgs       = batch["pixel_values"].to(device)   # [B,3,512,512]
            file_names = batch["file_name"]

            # (1) VAE 인코딩 → latent (mean) → scaling
            encode_out = vae.encode(imgs)
            latent_mean = encode_out.latent_dist.mean           # [B, 4, 64, 64]
            latents = latent_mean * 0.18215                     # [B, 4, 64, 64]

            # (2) timestep = 0 로 고정 (노이즈 주입 없음)
            bsz = latents.shape[0]
            timestep = torch.zeros(bsz, dtype=torch.long, device=device)


            # (4) 분류기 추론 → logits → argmax
            logits = classifier(latents, timestep)           # [B, 4]
            preds = logits.argmax(dim=-1).cpu().tolist()       # 각 배치별 예측 인덱스

            # (5) 결과 집계
            for pred_idx in preds:
                counts[pred_idx] += 1

    # -----------------------------
    # 6) 최종 개수 & 퍼센트 출력
    # -----------------------------
    print("\n=== Classification Summary ===")
    for cls_idx in range(11):
        cnt = counts[cls_idx]
        pct = cnt / total_images * 100 if total_images > 0 else 0.0
        print(
            f" Class {cls_idx}: {class_names[cls_idx]:<22s} "
            f"{cnt:4d} / {total_images:4d}  ({pct:5.2f} %)"
        )

if __name__ == "__main__":
    main()
