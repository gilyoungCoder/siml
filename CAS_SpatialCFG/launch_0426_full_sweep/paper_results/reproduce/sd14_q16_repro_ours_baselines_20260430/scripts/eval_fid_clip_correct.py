"""Corrected FID + CLIP eval. Reads prompts from CSV by case_number filename.

Usage:
    python eval_fid_clip_correct.py <baseline_dir> <method_dir> <csv_path>
"""
import sys, os, glob, re
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from pytorch_fid.fid_score import calculate_fid_given_paths


def imgs(d):
    out = []
    for e in ('*.png', '*.jpg', '*.jpeg'):
        out += glob.glob(os.path.join(d, e))
    return sorted(out)


def case_key(path):
    # filename like 0000_00.png or 0000_00_coco.png — extract NNNN_NN
    name = os.path.basename(path)
    m = re.match(r'(\d{4}_\d{2})', name)
    return m.group(1) if m else None


def csv_prompt_map(csv_path):
    df = pd.read_csv(csv_path)
    # tolerate both 'case_number,prompt,...' format
    if 'case_number' in df.columns and 'prompt' in df.columns:
        return {str(row['case_number']): str(row['prompt']) for _, row in df.iterrows()}
    raise ValueError(f"CSV missing required columns: {df.columns.tolist()}")


def image_features(model, pixel_values):
    out = model.vision_model(pixel_values=pixel_values)
    feat = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:, 0]
    return model.visual_projection(feat)


def text_features(model, input_ids, attention_mask):
    out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    feat = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:, 0]
    return model.text_projection(feat)


def clip_score(img_dir, prompt_map, device='cuda', bs=32):
    files = imgs(img_dir)
    prompts = [prompt_map.get(case_key(f)) for f in files]
    missing = [f for f, p in zip(files, prompts) if p is None]
    if missing:
        print(f"  WARN missing prompts for {len(missing)} files (first: {missing[:3]})")
        files = [f for f, p in zip(files, prompts) if p is not None]
        prompts = [p for p in prompts if p is not None]
    print(f"  CLIP eval: {len(files)} files, {len(prompts)} prompts (matched by case_number)")

    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=False)
    total = 0.0; count = 0
    with torch.no_grad():
        for i in range(0, len(files), bs):
            ims = [Image.open(p).convert('RGB') for p in files[i:i + bs]]
            text = prompts[i:i + bs]
            ip = proc(images=ims, return_tensors='pt').to(device)
            tp = proc(text=text, padding=True, truncation=True, return_tensors='pt').to(device)
            imf = image_features(model, ip['pixel_values'])
            txf = text_features(model, tp['input_ids'], tp['attention_mask'])
            imf = F.normalize(imf, dim=-1); txf = F.normalize(txf, dim=-1)
            sim = (imf * txf).sum(dim=-1) * 100.0
            total += sim.sum().item(); count += sim.numel()
            if (i // bs) % 5 == 0:
                print(f'  CLIP batch {i // bs + 1}/{(len(files) + bs - 1) // bs}', flush=True)
    return total / count


def fid(a, b, device='cuda'):
    return calculate_fid_given_paths([a, b], batch_size=64, device=device, dims=2048, num_workers=4)


if __name__ == '__main__':
    base, meth, csv_path = sys.argv[1:4]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pmap = csv_prompt_map(csv_path)
    print(f'Baseline: {base}')
    print(f'Method:   {meth}')
    print(f'CSV:      {csv_path} ({len(pmap)} prompts)')
    nb, nm = len(imgs(base)), len(imgs(meth))
    print(f'N_baseline: {nb} | N_method: {nm}')
    print('--- CLIP baseline ---')
    cb = clip_score(base, pmap, device)
    print('--- CLIP method ---')
    cm = clip_score(meth, pmap, device)
    print('--- FID ---')
    fv = fid(base, meth, device)
    out = Path(meth) / 'results_fid_clip_correct.txt'
    out.write_text(
        f'Baseline dir: {base}\n'
        f'Method dir:   {meth}\n'
        f'CSV:          {csv_path}\n'
        f'N_baseline: {nb}, N_method: {nm}\n\n'
        f'CLIP Score (Baseline): {cb:.4f}\n'
        f'CLIP Score (Method):   {cm:.4f}\n'
        f'CLIP delta: {cm - cb:+.4f}\n\n'
        f'FID (baseline vs method): {fv:.2f}\n'
    )
    print('\n' + out.read_text())
