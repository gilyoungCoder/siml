"""SLD-Max generation — standalone script"""
import torch, os, argparse, json, pandas as pd

def main(args):
    from diffusers.pipelines.stable_diffusion_safe import StableDiffusionPipelineSafe, SafetyConfig

    prompts = []
    if args.prompts.endswith('.csv'):
        df = pd.read_csv(args.prompts)
        col = 'prompt' if 'prompt' in df.columns else df.columns[0]
        prompts = [str(p) for p in df[col].tolist() if p is not None and str(p).strip()]
    else:
        with open(args.prompts) as f:
            prompts = [l.strip() for l in f if l.strip()]

    pipe = StableDiffusionPipelineSafe.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    cfg = SafetyConfig.MAX

    os.makedirs(args.outdir, exist_ok=True)
    for i, p in enumerate(prompts):
        op = os.path.join(args.outdir, f"{i:04d}_00.png")
        if os.path.exists(op):
            continue
        try:
            img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
                       generator=torch.Generator("cuda").manual_seed(42), **cfg).images[0]
            img.save(op)
        except Exception as e:
            print(f"  Error {i}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(prompts)}")

    n = len([x for x in os.listdir(args.outdir) if x.endswith('.png')])
    print(f"Done! {n} images")
    with open(os.path.join(args.outdir, "generation_stats.json"), "w") as f:
        json.dump({"method": "SLD-Max", "total": len(prompts), "generated": n}, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    main(p.parse_args())
