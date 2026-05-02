import argparse, time, json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prompts'); ap.add_argument('--outdir'); ap.add_argument('--steps',type=int,default=50); ap.add_argument('--cfg',type=float,default=7.5); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    prompts=[l.strip() for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    pipe=StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to('cuda')
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    times=[]
    for i,p in enumerate(prompts):
        gen=torch.Generator(device='cuda').manual_seed(args.seed+i)
        torch.cuda.synchronize(); t0=time.perf_counter()
        img=pipe(p, num_inference_steps=args.steps, guidance_scale=args.cfg, generator=gen).images[0]
        torch.cuda.synchronize(); dt=time.perf_counter()-t0
        print(f'Wall-Clock Time for image generation (Case#: {i:04d}_00): {dt:.4f} seconds', flush=True)
        img.save(out/f'{i:04d}_00.png'); times.append(dt)
    (out/'runtime_times.json').write_text(json.dumps({'method':'baseline_sd14','times':times,'mean':sum(times)/len(times)}, indent=2))
    print(f'RUNTIME_SUMMARY method=baseline_sd14 n={len(times)} mean={sum(times)/len(times):.4f}')
if __name__=='__main__': main()
