#!/usr/bin/env python3
# evaluate_11class.py

import argparse, os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser("Evaluate 11-class nudity classifier")
    p.add_argument("--not_people_data_path", required=True,
                   help="예: /.../sdxlLight/imagenet")
    p.add_argument("--classes10_dir", required=True,
                   help="예: /.../sdxlLight/10class")
    p.add_argument("--model_path", required=True,
                   help="학습된 classifier_final.pth 경로")
    p.add_argument("--sdxl_base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()

CLS_10 = [
    "fully_clothed","casual_wear","summer_casual","athletic_wear",
    "one_piece_swimwear","bikini_swimwear","lingerie",
    "topless_with_jeans","implied_nude","artistic_full_nude",
]

class ElevenDataset(Dataset):
    def __init__(self, not_people_dir, classes10_dir, transform=None):
        self.samples = []
        # 0: not_people
        for f in sorted(Path(not_people_dir).iterdir()):
            if f.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
                self.samples.append((str(f), 0))
        # 1–10: 각 서브폴더
        for idx, cls in enumerate(CLS_10, start=1):
            d = Path(classes10_dir)/cls
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
                    self.samples.append((str(f), idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}

def main():
    args = parse_args()
    device = torch.device(args.device)

    # DataLoader
    tfm = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    full_ds = ElevenDataset(args.not_people_data_path, args.classes10_dir, tfm)
    total = len(full_ds)
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # VAE & 4-step DDIM
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(device)
    vae.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    alphas = sched.alphas_cumprod.to(device)
    sqrt_a   = alphas.sqrt().view(1000,1,1,1)
    sqrt_1_a = (1 - alphas).sqrt().view(1000,1,1,1)

    # Classifier 로드
    clf = load_discriminator(
        ckpt_path=None, condition=None,
        eval=False, channel=4, num_classes=11
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    clf.load_state_dict(state)
    clf.eval()

    # 평가 루프
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            lat   = vae.encode(imgs).latent_dist.sample() * 0.18215
            bs    = lat.size(0)
            ts    = torch.randint(0,1000,(bs,),device=device)
            noise = torch.randn_like(lat)
            noisy = sqrt_a[ts]*lat + sqrt_1_a[ts]*noise
            logits = clf(noisy, ts.float()/1000)
            preds  = logits.argmax(-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # 매트릭스 계산
    cm = torch.zeros(11,11,dtype=torch.int64)
    for t,p in zip(y_true,y_pred):
        cm[t,p] += 1
    total = len(y_true)
    correct = sum(t==p for t,p in zip(y_true,y_pred))
    print(f"Overall accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm.numpy())

    print("\nPer-class accuracy:")
    for i in range(11):
        cnt = cm[i].sum().item()
        acc = cm[i,i].item()/cnt*100 if cnt>0 else 0.0
        name = "not_people" if i==0 else CLS_10[i-1]
        print(f" {i:2d} ({name:20s}): {acc:6.2f}% ({cm[i,i].item()}/{cnt})")

if __name__=="__main__":
    main()
