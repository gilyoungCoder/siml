"""
FID + CLIP score evaluation
Usage:
    python eval_fid_clip.py <baseline_dir> <method_dir> <prompts_txt>
"""
import sys, os, glob
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore
from pytorch_fid.fid_score import calculate_fid_given_paths

def load_prompts(txt_path, nsamples=4):
    """Load prompts, repeat nsamples times to match image count."""
    with open(txt_path) as f:
        prompts = [l.strip() for l in f if l.strip()]
    return [p for p in prompts for _ in range(nsamples)]

def get_images_sorted(img_dir):
    exts = ['*.png', '*.jpg', '*.jpeg']
    imgs = []
    for ext in exts:
        imgs += glob.glob(os.path.join(img_dir, ext))
    return sorted(imgs)

def compute_clip(img_dir, prompts, device='cuda', batch_size=32):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    imgs = get_images_sorted(img_dir)
    assert len(imgs) == len(prompts), f"img count {len(imgs)} != prompt count {len(prompts)}"

    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    scores = []
    for i in range(0, len(imgs), batch_size):
        batch_imgs = imgs[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        tensors = torch.stack([to_tensor(Image.open(p).convert('RGB')) for p in batch_imgs]).to(device)
        # CLIPScore expects uint8 [0,255]
        tensors_uint8 = (tensors * 255).to(torch.uint8)
        score = metric(tensors_uint8, batch_prompts)
        scores.append(score.item())
        if (i // batch_size) % 5 == 0:
            print(f"  CLIP batch {i//batch_size+1}/{(len(imgs)+batch_size-1)//batch_size} ...", flush=True)

    return sum(scores) / len(scores)

def compute_fid(dir1, dir2, device='cuda', batch_size=64, dims=2048):
    print(f"  Computing FID: {Path(dir1).name} vs {Path(dir2).name}", flush=True)
    fid = calculate_fid_given_paths(
        [dir1, dir2],
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=4,
    )
    return fid

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: eval_fid_clip.py <baseline_dir> <method_dir> <prompts_txt>")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    method_dir   = sys.argv[2]
    prompts_txt  = sys.argv[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(prompts_txt, nsamples=4)
    n_baseline = len(get_images_sorted(baseline_dir))
    n_method   = len(get_images_sorted(method_dir))
    print(f"Baseline: {n_baseline} images | Method: {n_method} images | Prompts: {len(prompts)}")

    # CLIP scores
    print("\n=== CLIP Score ===")
    print("Baseline CLIP...", flush=True)
    clip_base = compute_clip(baseline_dir, prompts[:n_baseline], device=device)
    print(f"  Baseline CLIP: {clip_base:.4f}")

    print("Method CLIP...", flush=True)
    clip_meth = compute_clip(method_dir, prompts[:n_method], device=device)
    print(f"  Method   CLIP: {clip_meth:.4f}")

    # FID
    print("\n=== FID Score ===")
    fid = compute_fid(baseline_dir, method_dir, device=device)
    print(f"  FID (baseline vs method): {fid:.2f}")

    # Save results
    out = Path(method_dir) / "results_fid_clip.txt"
    with open(out, 'w') as f:
        f.write(f"Baseline dir: {baseline_dir}\n")
        f.write(f"Method dir:   {method_dir}\n")
        f.write(f"N_baseline: {n_baseline}, N_method: {n_method}\n\n")
        f.write(f"CLIP Score (Baseline): {clip_base:.4f}\n")
        f.write(f"CLIP Score (Method):   {clip_meth:.4f}\n")
        f.write(f"CLIP delta: {clip_meth - clip_base:+.4f}\n\n")
        f.write(f"FID (baseline vs method): {fid:.2f}\n")
    print(f"\nSaved to {out}")
