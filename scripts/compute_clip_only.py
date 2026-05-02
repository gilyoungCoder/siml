#!/usr/bin/env python3
"""
Compute CLIP score only for final_coco dirs and merge into existing eval_metrics.json.
Usage:
    python scripts/compute_clip_only.py --dirs /path/to/dir1 /path/to/dir2 ...
    python scripts/compute_clip_only.py  # defaults to all final_coco subdirs
"""
import argparse
import json
import os
import sys
from pathlib import Path
from glob import glob
import multiprocessing as mp

import torch
from PIL import Image
from tqdm import tqdm

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
COCO_PROMPTS_PATH = "/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k_10k.csv"
BASE = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/final_coco"

ALL_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


def get_image_files(img_dir):
    files = []
    for ext in ALL_EXTS:
        files.extend(glob(os.path.join(img_dir, f'*{ext}')))
        files.extend(glob(os.path.join(img_dir, f'*{ext.upper()}')))
    return sorted(files)


def load_prompts(path):
    import csv
    prompts = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        prompt_col = 0
        for i, h in enumerate(header):
            if 'prompt' in h.lower():
                prompt_col = i
                break
        for row in reader:
            if row:
                prompts.append(row[prompt_col].strip())
    return prompts


def compute_clip_score(img_files, prompts, device, batch_size=64):
    from transformers import CLIPProcessor, CLIPModel

    n = min(len(img_files), len(prompts))
    img_files = img_files[:n]
    prompts = prompts[:n]

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()

    scores = []
    for i in tqdm(range(0, n, batch_size), desc="CLIP"):
        bf = img_files[i:i+batch_size]
        bp = prompts[i:i+batch_size]
        images = []
        valid_prompts = []
        for f, p in zip(bf, bp):
            try:
                images.append(Image.open(f).convert('RGB'))
                valid_prompts.append(p)
            except:
                continue
        if not images:
            continue
        inputs = processor(text=valid_prompts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image.diagonal().cpu().tolist()
        scores.extend([s / 100.0 for s in logits])

    del model, processor
    torch.cuda.empty_cache()
    return sum(scores) / len(scores) if scores else 0.0


def worker(gpu_id, img_dir, prompts, batch_size):
    """Single-GPU worker: compute CLIP score for one directory."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    name = os.path.basename(img_dir)
    device = torch.device("cuda:0")

    img_files = get_image_files(img_dir)
    if not img_files:
        print(f"[GPU {gpu_id}] [{name}] No images, skip")
        return

    metrics_path = os.path.join(img_dir, "eval_metrics.json")
    existing = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
        if "clip_score" in existing:
            print(f"[GPU {gpu_id}] [{name}] Already has clip_score={existing['clip_score']:.4f}, skip")
            return

    print(f"[GPU {gpu_id}] [{name}] Computing CLIP score for {len(img_files)} images...", flush=True)
    clip_score = compute_clip_score(img_files, prompts, device, batch_size)
    print(f"[GPU {gpu_id}] [{name}] CLIP score: {clip_score:.4f}", flush=True)

    existing["clip_score"] = clip_score
    with open(metrics_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"[GPU {gpu_id}] [{name}] Saved to {metrics_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.dirs is None:
        args.dirs = sorted(glob(os.path.join(BASE, "*/")))
        args.dirs = [d.rstrip("/") for d in args.dirs if os.path.isdir(d) and "logs" not in d]

    # Also include text_exit coco dir
    text_exit_coco = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/text_exit_20260202_184334/coco/mon0.05_gs12.5_bs2.0_sp0.2-0.3_txt0.50"
    if os.path.isdir(text_exit_coco) and text_exit_coco not in args.dirs:
        args.dirs.append(text_exit_coco)

    prompts = load_prompts(COCO_PROMPTS_PATH)
    print(f"Loaded {len(prompts)} prompts")
    print(f"Directories: {len(args.dirs)}")
    for d in args.dirs:
        print(f"  - {os.path.basename(d)}")
    print(f"GPUs: {args.num_gpus}")

    # Filter dirs that need processing
    todo_dirs = []
    for img_dir in args.dirs:
        metrics_path = os.path.join(img_dir, "eval_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                existing = json.load(f)
            if "clip_score" in existing:
                print(f"[{os.path.basename(img_dir)}] Already done, skip")
                continue
        todo_dirs.append(img_dir)

    if not todo_dirs:
        print("Nothing to compute!")
        return

    print(f"\nWill compute CLIP score for {len(todo_dirs)} dirs")

    # Launch processes, one per dir, assigned round-robin to GPUs
    gpu_list = list(range(args.num_gpus))
    procs = []
    for i, img_dir in enumerate(todo_dirs):
        gpu_id = gpu_list[i % len(gpu_list)]
        p = mp.Process(target=worker, args=(gpu_id, img_dir, prompts, args.batch_size))
        p.start()
        procs.append(p)

        # If all GPUs busy, wait for current batch
        if len(procs) >= len(gpu_list):
            for p in procs:
                p.join()
            procs = []

    for p in procs:
        p.join()

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
