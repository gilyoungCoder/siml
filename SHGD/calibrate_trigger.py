"""
Calibrate trigger threshold by measuring cosine similarity between
prompt predictions and harmful concept predictions.

Measures sim for RingABell (harmful) and COCO (safe) prompts to find
a threshold that separates them.
"""

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from pipeline_shgd import SHGDPipeline
import yaml
import argparse


def dummy(images, **kwargs):
    return images, [False] * len(images)


def measure_sim(pipe, prompt, harmful_emb, device, seed=42):
    """
    Measure directional cosine similarity between prompt and harmful concept.
    Compares (ε_cond - ε_uncond) vs (ε_harm - ε_uncond) to cancel noise.
    """
    device = torch.device(device) if isinstance(device, str) else device
    gen = torch.Generator(device=device).manual_seed(seed)

    # Encode prompt
    text_emb = pipe._encode_prompt(prompt, device, 1, True, None)

    # Prepare latents and timesteps
    pipe.scheduler.set_timesteps(50, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, 512, 512,
        text_emb.dtype, device, gen, None,
    )

    # First step (t=0, highest noise)
    t = timesteps[0]

    # Unconditional + conditional (text_emb = [uncond, cond])
    inp = torch.cat([latents] * 2)
    inp = pipe.scheduler.scale_model_input(inp, t)
    pred = pipe.unet(inp, t, encoder_hidden_states=text_emb).sample
    uncond_pred, cond_pred = pred.chunk(2)

    # Harmful prediction
    inp_s = pipe.scheduler.scale_model_input(latents, t)
    harm_pred = pipe.unet(inp_s, t, encoder_hidden_states=harmful_emb).sample

    # Direction vectors (cancel noise-dominant base)
    prompt_dir = cond_pred - uncond_pred
    harm_dir = harm_pred - uncond_pred

    sim = F.cosine_similarity(
        prompt_dir.reshape(1, -1),
        harm_dir.reshape(1, -1),
        dim=-1,
    ).item()
    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_harmful", type=int, default=20)
    parser.add_argument("--n_safe", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    model_cfg = config["model"]

    pipe = SHGDPipeline.from_pretrained(
        model_cfg["model_id"],
        scheduler=DDPMScheduler.from_pretrained(model_cfg["model_id"], subfolder="scheduler"),
        torch_dtype=torch.float32,
        revision="fp16",
    )
    pipe.safety_checker = dummy
    pipe = pipe.to(args.device)
    pipe.unet.eval()
    pipe.unet.requires_grad_(False)

    # Encode harmful concepts
    harmful_concepts = config.get("harmful_concepts", [
        "Sexual Acts", "Nudity", "Pornography",
    ])
    harmful_text = ", ".join(harmful_concepts)
    harmful_ids = pipe.tokenizer(
        harmful_text, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(args.device)
    harmful_emb = pipe.text_encoder(harmful_ids)[0]

    # Load prompts
    rab_prompts = []
    with open("../rab_grid_search/data/ringabell_full.txt") as f:
        for line in f:
            if line.strip():
                rab_prompts.append(line.strip())

    coco_prompts = []
    with open("../prompts/coco/coco_10k.txt") as f:
        for line in f:
            if line.strip():
                coco_prompts.append(line.strip())

    print("=" * 70)
    print("TRIGGER CALIBRATION")
    print("=" * 70)

    # Measure harmful prompts
    print(f"\n--- RingABell (harmful) prompts (first {args.n_harmful}) ---")
    harm_sims = []
    for i, p in enumerate(rab_prompts[:args.n_harmful]):
        sim = measure_sim(pipe, p, harmful_emb, args.device, seed=42 + i)
        harm_sims.append(sim)
        print(f"  [{i:3d}] sim={sim:.4f} | {p[:70]}")

    # Measure safe prompts
    print(f"\n--- COCO (safe) prompts (first {args.n_safe}) ---")
    safe_sims = []
    for i, p in enumerate(coco_prompts[:args.n_safe]):
        sim = measure_sim(pipe, p, harmful_emb, args.device, seed=42 + i)
        safe_sims.append(sim)
        print(f"  [{i:3d}] sim={sim:.4f} | {p[:70]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Harmful: mean={sum(harm_sims)/len(harm_sims):.4f}, "
          f"min={min(harm_sims):.4f}, max={max(harm_sims):.4f}")
    print(f"Safe:    mean={sum(safe_sims)/len(safe_sims):.4f}, "
          f"min={min(safe_sims):.4f}, max={max(safe_sims):.4f}")

    # Find threshold
    harm_min = min(harm_sims)
    safe_max = max(safe_sims)
    if harm_min > safe_max:
        threshold = (harm_min + safe_max) / 2
        print(f"\nClean separation! Suggested threshold: {threshold:.4f}")
        print(f"  (harmful min={harm_min:.4f} > safe max={safe_max:.4f})")
    else:
        # Find threshold that minimizes misclassification
        best_t, best_acc = 0, 0
        for t_cand in [i * 0.01 for i in range(-50, 100)]:
            t_cand = t_cand / 100.0 + 0.5  # range 0.0 to 1.0
            tp = sum(1 for s in harm_sims if s > t_cand)
            tn = sum(1 for s in safe_sims if s <= t_cand)
            acc = (tp + tn) / (len(harm_sims) + len(safe_sims))
            if acc > best_acc:
                best_acc = acc
                best_t = t_cand
        print(f"\nOverlap exists. Best threshold: {best_t:.4f} (accuracy={best_acc:.2%})")
        print(f"  Harmful above threshold: {sum(1 for s in harm_sims if s > best_t)}/{len(harm_sims)}")
        print(f"  Safe below threshold:    {sum(1 for s in safe_sims if s <= best_t)}/{len(safe_sims)}")


if __name__ == "__main__":
    main()
