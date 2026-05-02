import sys, os, glob, csv
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from pytorch_fid.fid_score import calculate_fid_given_paths

def imgs(d):
    out=[]
    for e in ('*.png','*.jpg','*.jpeg'): out += glob.glob(os.path.join(d,e))
    return sorted(out)

def load_prompts(path):
    p=Path(path)
    if p.suffix.lower()=='.csv':
        with p.open() as f:
            r=csv.DictReader(f)
            col=next((c for c in ['prompt','caption','recaption','sensitive prompt','adv_prompt','text'] if c in (r.fieldnames or [])), None)
            if col is None: raise SystemExit(f'no prompt col {r.fieldnames}')
            return [row[col].strip() for row in r if row.get(col,'').strip()]
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def image_features(model, pixel_values):
    out=model.vision_model(pixel_values=pixel_values)
    feat=out.pooler_output if getattr(out,'pooler_output',None) is not None else out.last_hidden_state[:,0]
    return model.visual_projection(feat)

def text_features(model, input_ids, attention_mask):
    out=model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    feat=out.pooler_output if getattr(out,'pooler_output',None) is not None else out.last_hidden_state[:,0]
    return model.text_projection(feat)

def clip_score(img_dir, prompts, device='cuda', bs=32):
    model=CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
    proc=CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=False)
    files=imgs(img_dir); assert len(files)==len(prompts), f'{img_dir}: {len(files)} imgs != {len(prompts)} prompts'
    total=0.0; count=0
    with torch.no_grad():
        for i in range(0,len(files),bs):
            ims=[Image.open(p).convert('RGB') for p in files[i:i+bs]]; text=prompts[i:i+bs]
            ip=proc(images=ims, return_tensors='pt').to(device)
            tp=proc(text=text, padding=True, truncation=True, return_tensors='pt').to(device)
            imf=F.normalize(image_features(model, ip['pixel_values']), dim=-1)
            txf=F.normalize(text_features(model, tp['input_ids'], tp['attention_mask']), dim=-1)
            total += ((imf*txf).sum(dim=-1)*100.0).sum().item(); count += len(ims)
            if (i//bs)%20==0: print(f'  CLIP batch {i//bs+1}/{(len(files)+bs-1)//bs}', flush=True)
    return total/count

def fid(a,b,device='cuda'):
    return calculate_fid_given_paths([a,b], batch_size=64, device=device, dims=2048, num_workers=4)

if __name__=='__main__':
    base,meth,prompt_path,out_path=sys.argv[1:5]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    prompts=load_prompts(prompt_path)
    nb=len(imgs(base)); nm=len(imgs(meth))
    assert nb==nm==len(prompts), (nb,nm,len(prompts))
    print(f'Baseline: {nb} | Method: {nm} | Prompts: {len(prompts)}')
    cb=clip_score(base,prompts,device); cm=clip_score(meth,prompts,device); fv=fid(base,meth,device)
    text=f'Baseline dir: {base}\nMethod dir:   {meth}\nPrompt file:  {prompt_path}\nN_baseline: {nb}, N_method: {nm}\n\nCLIP Score (Baseline): {cb:.4f}\nCLIP Score (Method):   {cm:.4f}\nCLIP delta: {cm-cb:+.4f}\n\nFID (baseline vs method): {fv:.2f}\n'
    Path(out_path).write_text(text)
    Path(meth,'results_fid_clip_coco10k.txt').write_text(text)
    print(text)
