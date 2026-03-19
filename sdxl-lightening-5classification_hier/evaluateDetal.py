# evaluate_classifier.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator, discriminator_defaults
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from PIL import Image

class ThreeClassDataset(Dataset):
    """
    3-class dataset:
      0: 사람 없음 (benign)
      1: 사람 있음 (person)
      2: 사람 누드 (nude)
    """
    def __init__(self, benign_dir, person_dir, nude_dir, transform=None):
        self.paths = []
        self.labels = []
        for fname in sorted(os.listdir(benign_dir)):
            self.paths.append(os.path.join(benign_dir, fname))
            self.labels.append(0)
        for fname in sorted(os.listdir(person_dir)):
            self.paths.append(os.path.join(person_dir, fname))
            self.labels.append(1)
        for fname in sorted(os.listdir(nude_dir)):
            self.paths.append(os.path.join(nude_dir, fname))
            self.labels.append(2)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return {"pixel_values": img, "label": label}

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="3-class classifier evaluation with DDPM noise injection")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="VAE & scheduler가 들어 있는 Stable Diffusion 체크포인트 폴더")
    p.add_argument("--classifier_ckpt", type=str, required=True,
                   help="훈련된 분류기 .pth 파일 경로")
    p.add_argument("--benign_dir", type=str, required=True, help="benign 이미지 디렉토리")
    p.add_argument("--person_dir", type=str, required=True, help="person 이미지 디렉토리")
    p.add_argument("--nude_dir", type=str, required=True, help="nude 이미지 디렉토리")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42, help="validation용 random seed")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터셋 & split (train/val 90/10, validation 부분만 사용)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    full_ds = ThreeClassDataset(
        benign_dir=args.benign_dir,
        person_dir=args.person_dir,
        nude_dir=args.nude_dir,
        transform=transform
    )
    total = len(full_ds)
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2) VAE & DDPM 스케줄러 로드
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 3) 분류기 로드 (eval 모드)
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,      # VAE latent 채널 수
        num_classes=3   # 3-way 분류
    ).to(device)
    classifier.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    # [Modified] 4-0) confusion matrix 초기화
    num_classes = 3                                          # [Modified]
    conf_mat = [[0]*num_classes for _ in range(num_classes)] # [Modified]

    # 4) Validation loop (훈련 스크립트와 똑같이)
    correct = [0,0,0]
    total_cls = [0,0,0]
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)             # [B]

            # 4-1) VAE encode → latent
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215  # [B,4,H/8,W/8]

            bsz = lat.shape[0]
            # 4-2) 랜덤 timestep & 노이즈
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long
            )                                               # [B]
            noise = torch.randn_like(lat)                   # [B,4,H/8,W/8]

            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(lat.ndim-1)))
            noisy_lat = torch.sqrt(alpha_bar) * lat + torch.sqrt(1-alpha_bar) * noise

            # 4-3) 분류기 forward + loss
            logits = classifier(noisy_lat, timesteps)       # [B,3]
            loss   = loss_fn(logits, labels)
            running_loss += loss.item() * bsz

            preds = logits.argmax(dim=-1)
            # [Modified] detailed update: confusion matrix 포함
            labels_cpu = labels.cpu().tolist()              # [Modified]
            preds_cpu  = preds.cpu().tolist()               # [Modified]
            for gt, pr in zip(labels_cpu, preds_cpu):       # [Modified]
                total_cls[gt] += 1                          # (기존) 클래스별 총 샘플 수
                if gt == pr:
                    correct[gt] += 1                       # (기존) 클래스별 정답 수
                conf_mat[gt][pr] += 1                      # [Modified]
            total += bsz

    # 5) 결과 출력
    avg_loss = running_loss / total
    overall_acc = sum(correct) / total
    print(f"\nValidation Loss: {avg_loss:.4f}, Overall Acc: {overall_acc*100:.2f}% ({sum(correct)}/{total})")
    for cls in range(3):
        acc = correct[cls] / total_cls[cls] if total_cls[cls] > 0 else 0.0
        print(f"  Class {cls} accuracy: {acc*100:5.2f}%  ({correct[cls]}/{total_cls[cls]})")

    # [Modified] 6) 상세 오분류 통계 출력
    print("\n-- 상세 오분류 통계 (confusion matrix 기반) --")
    for gt in range(num_classes):                        # [Modified]
        total_gt = sum(conf_mat[gt])                     # [Modified]
        for pr in range(num_classes):                    # [Modified]
            rate = conf_mat[gt][pr] / total_gt if total_gt>0 else 0.0  # [Modified]
            print(f"P(gt={gt} → pred={pr}): {rate*100:5.2f}%  ({conf_mat[gt][pr]}/{total_gt})")  # [Modified]

if __name__ == "__main__":
    main()
