# classify_folder.py

import os
import argparse
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL
from geo_models.classifier.classifier import load_discriminator
from accelerate.utils import set_seed


# 11개 클래스 이름 매핑 (index → 이름)
CLASS_NAMES = {
    0: "Not People",
    1: "fully_clothed",
    2: "nude",
}


class ImageFolderDataset(Dataset):
    """
    주어진 폴더 내의 모든 이미지를 읽어서 반환합니다.
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.paths = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.paths.append(os.path.join(folder_path, fname))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
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
        description="폴더 내 이미지들을 11-클래스 분류기로 예측"
    )
    p.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Stable Diffusion 체크포인트 경로 (VAE & scheduler 포함)"
    )
    p.add_argument(
        "--classifier_ckpt", type=str, required=True,
        help="학습된 11-클래스 분류기 .pth 파일 경로"
    )
    p.add_argument(
        "--image_dir", type=str, required=True,
        help="분류를 수행할 이미지들이 들어있는 폴더 경로"
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="추론 시 배치 크기 (기본: 16)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="시드 설정 (기본: 42)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # 1) VAE 로드 (eval 모드)
    # ------------------------
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # -----------------------------------
    # 2) 11-클래스 분류기 로드 (eval 모드)
    # -----------------------------------
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    ).to(device)
    classifier.eval()

    # -------------------------------
    # 3) 이미지 폴더 데이터셋 구성
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolderDataset(
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
    print(f"Found {len(dataset)} images under {args.image_dir}")

    # -----------------------------------------------
    # 4) 노이즈 없이(timestep=0) 모든 이미지에 대해 예측
    # -----------------------------------------------
    results = []  # (file_name, pred_idx, pred_name) 저장용

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs        = batch["pixel_values"].to(device)   # [B,3,512,512]
            file_names  = batch["file_name"]

            # (1) VAE Encoding: latent_dist.mean 사용
            #     - AutoencoderKL.encode(...) 반환 객체의 latent_dist.mean
            #     - 상수 0.18215 곱해서 분류기에 들어갈 latent 얻음
            # ------------------------------------------------
            with torch.no_grad():
                encode_out = vae.encode(imgs)
                # latent_dist.mean: [B, 4, H/8, W/8]
                latent_mean = encode_out.latent_dist.mean
                latents = latent_mean * 0.18215

            # (2) timestep = 0 → 정규화된 timestep도 0
            # ------------------------------------------------
            bsz = latents.shape[0]
            norm_ts = torch.zeros(bsz, device=device)

            # (3) 분류기 forward → [B, 11] logits
            # ------------------------------------------------
            logits = classifier(latents, norm_ts)  # [B,11]

            # (4) argmax로 예측 인덱스 얻음
            preds_idx = logits.argmax(dim=-1).cpu().tolist()

            # (5) 결과 모아두기
            for fname, idx in zip(file_names, preds_idx):
                results.append((fname, idx, CLASS_NAMES[idx]))

    # -----------------
    # 5) 결과 출력
    # -----------------
    print("\n=== Classification Results ===")
    for fname, idx, cname in results:
        print(f"{fname:30s} → Predicted class: {idx} ({cname})")


if __name__ == "__main__":
    main()
