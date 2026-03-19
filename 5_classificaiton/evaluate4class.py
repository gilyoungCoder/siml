# evaluate_4class.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from PIL import Image

###############################################################################
# Dataset
###############################################################################
class FourClassDataset(Dataset):
    """
    4-class dataset
      0: Nonpeople
      1: Clothed          (Safe)
      2: Revealing (Non-nude)  (Partial)
      3: Full Nudity      (Full)
    """
    def __init__(
        self,
        nonpeople_dir: str,
        clothed_dir: str,
        revealing_dir: str,
        full_nudity_dir: str,
        transform=None,
    ):
        self.paths, self.labels = [], []
        dirs = [
            (nonpeople_dir, 0),
            (clothed_dir,   1),
            (revealing_dir, 2),
            (full_nudity_dir, 3),
        ]
        for d, label in dirs:
            if not os.path.isdir(d):
                raise ValueError(f"디렉토리 경로를 확인하십시오: {d}")
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(d, fname))
                    self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx]}

###############################################################################
# Argument parser
###############################################################################
def parse_args():
    p = argparse.ArgumentParser(
        description="4-class classifier evaluation with DDPM noise injection"
    )
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="Stable Diffusion 체크포인트 경로(VAE & scheduler 포함)")
    p.add_argument("--classifier_ckpt", type=str, required=True,
                   help="학습된 분류기 .pth 경로")
    # 클래스별 디렉토리
    p.add_argument("--nonpeople_dir",   type=str, required=True)
    p.add_argument("--clothed_dir",     type=str, required=True)
    p.add_argument("--revealing_dir",   type=str, required=True)
    p.add_argument("--full_nudity_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

###############################################################################
# Main
###############################################################################
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ────────────────────── 데이터셋 ──────────────────────
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    full_ds = FourClassDataset(
        args.nonpeople_dir, args.clothed_dir,
        args.revealing_dir, args.full_nudity_dir,
        transform=tfm
    )
    val_size  = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    _, val_ds = random_split(full_ds, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # ─────────────────── VAE & Scheduler ──────────────────
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder="vae").to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ───────────────────── 분류기 ─────────────────────
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=4
    ).to(device)
    classifier.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    # Validation loop
    correct = [0]*4
    total_cls = [0]*4
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            # VAE encode & noise injection
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bsz = lat.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(lat)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(lat.ndim-1)))
            noisy_lat = torch.sqrt(alpha_bar) * lat + torch.sqrt(1 - alpha_bar) * noise

            # forward + loss
            norm_ts = timesteps / scheduler.config.num_train_timesteps
            logits = classifier(noisy_lat, norm_ts)  # [B,11]
            loss = loss_fn(logits, labels)
            running_loss += loss.item() * bsz

            preds = logits.argmax(dim=-1)
            for gt, pr in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                total_cls[gt] += 1
                if gt == pr:
                    correct[gt] += 1
            total += bsz

    # ─────────────────── 결과 출력 ────────────────────
    avg_loss = running_loss / total
    overall_acc = sum(correct) / total * 100
    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_acc:5.2f}% ({sum(correct)}/{total})\n")

    label_names = ["Safe (Clothed)", "Partial (Revealing)",
                   "Full Nudity", "Not People"]
    order       = [1, 2, 3, 0]  # safe, partial, full, not-people
    print("Per-class accuracy (order: safe / partial / full / not-people):")
    for idx in order:
        acc = correct[idx] / total_cls[idx] * 100 if total_cls[idx] else 0.0
        print(f"  {label_names[idx]:17s}: {acc:5.2f}%  ({correct[idx]}/{total_cls[idx]})")

###############################################################################
if __name__ == "__main__":
    main()
