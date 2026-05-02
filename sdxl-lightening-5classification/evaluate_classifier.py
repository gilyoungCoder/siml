#!/usr/bin/env python3
# evaluate_4class.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from PIL import Image

class FourClassDataset(Dataset):
    """0: not_people | 1: fully_clothed | 2: partial_nude | 3: full_nude"""
    def __init__(self,
                 not_people_dir: str,
                 fully_clothed_dir: str,
                 partial_nude_dir: str,
                 full_nude_dir: str,
                 transform=None):
        self.samples = []
        for root, label in [
            (not_people_dir,    0),
            (fully_clothed_dir, 1),
            (partial_nude_dir,  2),
            (full_nude_dir,     3),
        ]:
            if not os.path.isdir(root):
                raise ValueError(f"디렉터리를 찾을 수 없습니다: {root}")
            for fname in sorted(os.listdir(root)):
                if fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                    self.samples.append((os.path.join(root, fname), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="4-class classifier evaluation with DDPM noise injection"
    )
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="Stable Diffusion VAE & scheduler checkpoint 경로")
    p.add_argument("--classifier_ckpt", type=str, required=True,
                   help="훈련된 분류기 .pth 파일 경로")
    p.add_argument("--not_people_dir",    type=str, required=True,
                   help="0: 사람 없음 디렉토리")
    p.add_argument("--fully_clothed_dir", type=str, required=True,
                   help="1: fully-clothed 디렉토리")
    p.add_argument("--partial_nude_dir",  type=str, required=True,
                   help="2: partial-nude 디렉토리")
    p.add_argument("--full_nude_dir",     type=str, required=True,
                   help="3: full-nude 디렉토리")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed",       type=int, default=42,
                   help="random seed (validation split용)")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터셋 & 90/10 split (validation만 사용)
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    full_ds = FourClassDataset(
        not_people_dir    = args.not_people_dir,
        fully_clothed_dir = args.fully_clothed_dir,
        partial_nude_dir  = args.partial_nude_dir,
        full_nude_dir     = args.full_nude_dir,
        transform=transform
    )
    total = len(full_ds)
    val_size   = int(0.1 * total)
    train_size = total - val_size
    _, val_ds  = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )

    # 2) VAE & DDPM 스케줄러 로드
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)

    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # 3) 분류기 로드 (eval 모드)
    classifier = load_discriminator(
        ckpt_path    = args.classifier_ckpt,
        condition    = None,
        eval         = True,
        channel      = 4,
        num_classes  = 4
    ).to(device)
    classifier.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4) Validation loop
    correct   = [0]*4
    total_cls = [0]*4
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)           # [B]

            # 4-1) VAE encode → latent
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215  # [B,4,h,w]

            # 4-2) 랜덤 timestep & 노이즈
            bsz = lat.size(0)
            ts  = torch.randint(
                low=0,
                high=scheduler.num_train_timesteps,
                size=(bsz,),
                device=device,
            )
            noise = torch.randn_like(lat)

            alphas_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alphas_cumprod[ts].view(bsz,1,1,1)
            noisy_lat = torch.sqrt(alpha_bar) * lat + torch.sqrt(1-alpha_bar) * noise

            # 4-3) 분류기 forward + loss
            logits = classifier(noisy_lat, ts.float()/1000.0)  # [B,4]
            loss   = loss_fn(logits, labels)
            running_loss += loss.item() * bsz

            preds = logits.argmax(dim=-1)
            for gt, pr in zip(labels.cpu(), preds.cpu()):
                total_cls[gt] += 1
                if pr == gt:
                    correct[gt] += 1
            total += bsz

    # 5) 결과 출력
    avg_loss    = running_loss / total
    overall_acc = sum(correct) / total

    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_acc*100:.2f}% ({sum(correct)}/{total})")
    for cls in range(4):
        acc = correct[cls] / total_cls[cls] if total_cls[cls]>0 else 0.0
        print(f"  Class {cls} accuracy: {acc*100:5.2f}%  ({correct[cls]}/{total_cls[cls]})")

if __name__ == "__main__":
    main()
