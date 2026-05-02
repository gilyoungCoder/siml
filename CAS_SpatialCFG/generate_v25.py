#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v25: CLS × Scale approach — simplest image embedding method.

CLIP image CLS [768, norm=1] × scale → UNet conditioning → eps_img_target.
Uses v25 .pt files with pre-scaled embeddings.

Supports: --example_mode text / img / both
          --img_key target_scale{25,30,35,40}
"""

import os, sys, json, math, random, csv
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler


class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold, self.sticky, self.triggered = threshold, sticky, False
    def reset(self): self.triggered = False
    def compute(self, ep, en, et):
        dp = (ep-en).reshape(1,-1).float(); dt = (et-en).reshape(1,-1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
        if math.isnan(c) or math.isinf(c): return 0., self.triggered if self.sticky else False
        if self.sticky and self.triggered: return c, True
        if c > self.threshold:
            if self.sticky: self.triggered = True
            return c, True
        return c, False

def spatial_cas(ep, en, et, nbr=3):
    dp=(ep-en).float(); dt=(et-en).float()
    H,W=dp.shape[2],dp.shape[3]; p=nbr//2
    pu=F.unfold(dp,nbr,padding=p); tu=F.unfold(dt,nbr,padding=p)
    return F.cosine_similarity(pu,tu,dim=1).reshape(H,W)

def gauss_blur(x, ks=5, s=1.):
    co=torch.arange(ks,dtype=x.dtype,device=x.device)-ks//2
    g=torch.exp(-.5*(co/s)**2); g=g/g.sum()
    kh,kw=g.view(1,1,ks,1),g.view(1,1,1,ks); p=ks//2
    x=F.pad(x,[0,0,p,p],'reflect'); x=F.conv2d(x,kh.expand(x.shape[1],-1,-1,-1),groups=x.shape[1])
    x=F.pad(x,[p,p,0,0],'reflect'); return F.conv2d(x,kw.expand(x.shape[1],-1,-1,-1),groups=x.shape[1])

def soft_mask(raw, thr, alpha=10., blur=1.):
    m=torch.sigmoid(alpha*(raw-thr)).unsqueeze(0).unsqueeze(0)
    if blur>0: m=gauss_blur(m,s=blur)
    return m.clamp(0,1)

def load_prompts(fp):
    fp=Path(fp)
    if fp.suffix==".csv":
        ps=[]
        with open(fp) as f:
            r=csv.DictReader(f)
            col=next((c for c in ['sensitive prompt','adv_prompt','prompt','target_prompt','text','Prompt','Text'] if c in r.fieldnames),None)
            for row in r:
                p=row[col].strip()
                if p: ps.append(p)
        return ps
    return [l.strip() for l in open(fp) if l.strip()]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def encode_concepts(te, tok, concepts, dev):
    es=[]
    for c in concepts:
        inp=tok(c,padding="max_length",max_length=tok.model_max_length,truncation=True,return_tensors="pt")
        es.append(te(inp.input_ids.to(dev))[0])
    return torch.stack(es).mean(0)

def parse_args():
    p=ArgumentParser(description="v25: CLS×Scale safe generation")
    p.add_argument("--ckpt",default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts",required=True); p.add_argument("--outdir",required=True)
    p.add_argument("--nsamples",type=int,default=1); p.add_argument("--steps",type=int,default=50)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--cfg_scale",type=float,default=7.5)
    p.add_argument("--start_idx",type=int,default=0); p.add_argument("--end_idx",type=int,default=-1)
    p.add_argument("--cas_threshold",type=float,default=0.6)
    p.add_argument("--spatial_threshold",type=float,default=0.3)
    p.add_argument("--sigmoid_alpha",type=float,default=10.)
    p.add_argument("--neighborhood_size",type=int,default=3)
    p.add_argument("--blur_sigma",type=float,default=1.)
    p.add_argument("--safety_scale",type=float,default=1.0)
    p.add_argument("--example_mode",default="both",choices=["text","img","both"])
    p.add_argument("--fusion",default="union",choices=["union","soft_union","mean"])
    p.add_argument("--clip_file",default=None,help="v25 .pt file")
    p.add_argument("--img_key",default="target_scale35",help="Key in .pt for image embeds")
    p.add_argument("--target_concepts",nargs="+",default=["nudity","nude person","naked body"])
    p.add_argument("--anchor_concepts",nargs="+",default=["clothed person","person wearing clothes"])
    p.add_argument("--debug",action="store_true")
    a=p.parse_args()
    if a.example_mode in ("img","both") and a.clip_file is None:
        p.error("--clip_file required for img/both mode")
    return a

def main():
    args=parse_args(); set_seed(args.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_img = args.example_mode in ("img","both")

    print(f"v25: example={args.example_mode} fusion={args.fusion} ss={args.safety_scale} st={args.spatial_threshold} img_key={args.img_key}")

    prompts=load_prompts(args.prompts)
    end=args.end_idx if args.end_idx>0 else len(prompts)
    pw=list(enumerate(prompts))[args.start_idx:end]

    pipe=StableDiffusionPipeline.from_pretrained(args.ckpt,torch_dtype=torch.float16,safety_checker=None).to(dev)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config); pipe.feature_extractor=None
    unet,vae,tok,te,sched=pipe.unet,pipe.vae,pipe.tokenizer,pipe.text_encoder,pipe.scheduler

    with torch.no_grad():
        text_tgt=encode_concepts(te,tok,args.target_concepts,dev)
        anchor_emb=encode_concepts(te,tok,args.anchor_concepts,dev)
        unc=te(tok("",padding="max_length",max_length=tok.model_max_length,truncation=True,return_tensors="pt").input_ids.to(dev))[0]

    img_tgt=None
    if use_img:
        data=torch.load(args.clip_file,map_location="cpu")
        img_tgt=data[args.img_key].to(device=dev,dtype=next(unet.parameters()).dtype)
        print(f"  Image embeds: {args.img_key}, norm={img_tgt.float().norm(dim=-1).mean():.1f}")

    cas=GlobalCAS(args.cas_threshold)
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    stats=[]

    for pi,prompt in tqdm(pw,desc="Gen"):
        if not prompt.strip(): continue
        for si in range(args.nsamples):
            seed=args.seed+pi*args.nsamples+si; set_seed(seed); cas.reset()
            gc,cvs,mas=0,[],[]
            with torch.no_grad():
                pemb=te(tok(prompt,padding="max_length",max_length=tok.model_max_length,truncation=True,return_tensors="pt").input_ids.to(dev))[0]
            set_seed(seed)
            lat=torch.randn(1,4,64,64,device=dev,dtype=torch.float16)*sched.init_noise_sigma
            sched.set_timesteps(args.steps,device=dev)

            for si_t,t in enumerate(sched.timesteps):
                li=sched.scale_model_input(lat,t)
                with torch.no_grad():
                    raw=unet(torch.cat([li,li]),t,encoder_hidden_states=torch.cat([unc,pemb])).sample
                    en,ep=raw.chunk(2)
                    et=unet(li,t,encoder_hidden_states=text_tgt).sample
                ec=en+args.cfg_scale*(ep-en)
                cv,trig=cas.compute(ep,en,et); cvs.append(cv)

                if trig:
                    with torch.no_grad():
                        ea=unet(li,t,encoder_hidden_states=anchor_emb).sample

                    # Text mask
                    tmask=None
                    if args.example_mode in ("text","both"):
                        tmask=soft_mask(spatial_cas(ep,en,et,args.neighborhood_size).to(dev),
                                       args.spatial_threshold,args.sigmoid_alpha,args.blur_sigma)
                    # Image mask
                    imask=None
                    if use_img:
                        with torch.no_grad():
                            ei=unet(li,t,encoder_hidden_states=img_tgt).sample
                        imask=soft_mask(spatial_cas(ep,en,ei,args.neighborhood_size).to(dev),
                                       args.spatial_threshold,args.sigmoid_alpha,args.blur_sigma)
                    # Fuse
                    if args.example_mode=="text": fm=tmask
                    elif args.example_mode=="img": fm=imask
                    else:
                        if args.fusion=="union": fm=torch.max(tmask,imask)
                        elif args.fusion=="soft_union": fm=1-(1-tmask)*(1-imask)
                        else: fm=(tmask+imask)/2

                    # Anchor inpaint
                    m=fm.to(ec.dtype); blend=(args.safety_scale*m).clamp(max=1.)
                    ea_cfg=en+args.cfg_scale*(ea-en)
                    eps_final=ec*(1-blend)+ea_cfg*blend
                    if torch.isnan(eps_final).any(): eps_final=torch.where(torch.isfinite(eps_final),eps_final,ec)
                    gc+=1; mas.append(float(fm.mean()))

                    if args.debug and si_t%10==0:
                        ta=float(tmask.mean()) if tmask is not None else 0
                        ia=float(imask.mean()) if imask is not None else 0
                        print(f"  [{si_t:02d}] CAS={cv:.3f} t={ta:.3f} i={ia:.3f} f={mas[-1]:.3f}")
                else:
                    eps_final=ec

                lp=lat.clone(); lat=sched.step(eps_final,t,lat).prev_sample
                if torch.isnan(lat).any(): lat=sched.step(en+args.cfg_scale*(ep-en),t,lp).prev_sample

            with torch.no_grad():
                dec=vae.decode(lat.to(vae.dtype)/vae.config.scaling_factor).sample
                dec=(dec/2+.5).clamp(0,1)
                img=(dec[0].cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
            fn=f"{pi:04d}_{si:02d}.png"
            Image.fromarray(img).resize((512,512)).save(str(outdir/fn))
            stats.append({"pi":pi,"si":si,"seed":seed,"guided":gc,"max_cas":max(cvs) if cvs else 0})

    json.dump(stats,open(outdir/"generation_stats.json","w"),indent=2)
    json.dump(vars(args),open(outdir/"args.json","w"),indent=2)
    gi=sum(1 for s in stats if s["guided"]>0)
    print(f"Done! {len(stats)} imgs, guided {gi}/{len(stats)}")

if __name__=="__main__": main()
