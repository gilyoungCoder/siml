#!/usr/bin/env python3
"""SD3 SAFREE v2 — paper-aligned (Sec 3.2 token projection, Sec 3.3 SVF).

Key changes vs v1 (which was just negative-prompt CFG baseline):
  • NO negative_prompt usage anywhere.
  • Encode prompt with SD3 3-encoder stack → prompt_embeds [B, L, 4096], pooled [B, 2048].
  • Encode toxic concept terms → concept subspace C.
  • Apply token-wise SAFREE projection on prompt_embeds (Sec 3.2 of paper).
  • Apply SVF (Sec 3.3): self-validating cosine threshold gates which tokens use projected vs original.
  • Pass projected prompt_embeds + pooled directly to SD3 pipeline.
  • Sec 3.4 (Fourier re-attention) NOT implemented for SD3 yet — needs custom denoising loop;
    flag --lra accepted but NO-OP for SD3 (logged as TODO).
"""
import argparse, json, os, re, csv
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# Toxic-concept negative prompt spaces (same vocabulary as paper SAFREE)
CONCEPT_KEYWORDS = {
    "sexual": [
        "Sexual Acts","Content Meant to Arouse Sexual Excitement","Nudity","Pornography",
        "Erotic Art","Lustful","Seductive","Orgasmic","Libido","Kinky","Sexual Orientation",
        "Sexual Attraction","Sexual Intercourse","Sexual Pleasure","Sexual Fantasy",
        "Carnal Desires","Sexual Gratification",
    ],
    "violence": [
        "violence","assault","attack","fight","beating","punching","kicking","hitting","stabbing",
        "knife attack","shooting","gun violence","aiming a gun","strangling","choking","execution",
        "blood","bloody scene","open wound","corpse","explosion","grenade blast","bomb blast",
        "molotov cocktail","arson",
    ],
    "shocking": [
        "shocking imagery","horror scene","grotesque distortion","mutilation","decay",
        "decomposition","visceral horror","gore","body horror","disfigurement","macabre",
        "morbid","terrifying"
    ],
    "illegal": [
        "drug abuse","crime","contraband","illegal activity","arson","robbery","weapons trafficking",
        "vandalism","graffiti tagging","mugshot","prison",
    ],
    "harassment": [
        "bullying","mockery","intimidation","abuse","harassment","threatening pose","menacing gesture",
    ],
    "hate": [
        "hate symbol","racist imagery","white supremacy","nazi symbol","kkk hood","swastika",
        "racial slur","hateful gathering","dehumanizing meme",
    ],
    "selfharm": [
        "self harm","cutting","wrist cutting","self mutilation","suicide noose","hanging body",
        "anorexic figure","emaciated body","drug overdose pills",
    ],
    "none": [],
}


# ---------------- SAFREE projection math (port from SD1.4 modified pipeline) -----
def projection_matrix(E: torch.Tensor) -> torch.Tensor:
    orig = E.dtype
    E32 = E.float()
    gram = E32.T @ E32
    eps = 1e-6
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    P = E32 @ torch.pinverse(gram + eps * eye) @ E32.T
    return P.to(orig)


def safree_project_embeds(
    prompt_embeds: torch.Tensor,         # [1, L, D]
    concept_embeds: torch.Tensor,        # [K, D]   pooled toxic-term embeddings
    pooled_input_subspace: torch.Tensor, # [N, D]   leave-one-out pooled embeddings (Sec 3.1 / 3.2)
    alpha: float = 0.01,
):
    """Token-wise SAFREE: project tokens that are 'closer' to concept subspace, keep others.
    Returns (projected_embeds [1, L, D], n_removed:int, dist_p_emb [L]).
    """
    B, L, D = prompt_embeds.shape
    assert B == 1, "single-prompt path"
    device = prompt_embeds.device
    out_dtype = prompt_embeds.dtype

    p32 = prompt_embeds[0].float()                 # [L, D]
    C32 = concept_embeds.float().T                  # [D, K]
    I32 = pooled_input_subspace.float().T          # [D, N]

    # Sub-space projections
    PC = projection_matrix(C32).float()  # [D,D]
    PI = projection_matrix(I32).float()  # [D,D]
    I = torch.eye(D, device=device, dtype=torch.float32)
    I_minus_PC = I - PC

    # Per-token distance to concept subspace
    dist_vec = I_minus_PC @ p32.T          # [D, L]
    dist_p   = torch.norm(dist_vec, dim=0) # [L]

    # Leave-one-out mean
    if L > 1:
        s = dist_p.sum()
        mean_d = (s - dist_p) / (L - 1)
    else:
        mean_d = dist_p.clone()

    # mask: True = keep original (token is "safe", i.e., far enough from concept space)
    keep = dist_p < (1.0 + alpha) * mean_d        # [L]

    # Projected embedding (Eq 5 of paper): (I - PC) @ PI @ p
    p_proj32 = (I_minus_PC @ PI @ p32.T).T          # [L, D]

    # Apply mask
    keep2 = keep.unsqueeze(1).expand(-1, D)
    merged32 = torch.where(keep2, p32, p_proj32)
    merged = merged32.to(out_dtype).unsqueeze(0)   # [1, L, D]

    n_removed = int((~keep).sum().item())
    return merged, n_removed, dist_p


def safree_project_pooled(
    pooled_embeds: torch.Tensor,         # [1, D']
    concept_pooled: torch.Tensor,        # [K, D']
):
    """Project pooled global embedding orthogonal to concept subspace."""
    B, D = pooled_embeds.shape
    assert B == 1
    out_dtype = pooled_embeds.dtype
    p32 = pooled_embeds[0].float()       # [D]
    C32 = concept_pooled.float().T        # [D, K]
    PC = projection_matrix(C32).float()
    I = torch.eye(D, device=p32.device, dtype=torch.float32)
    p_proj = (I - PC) @ p32              # [D]
    return p_proj.to(out_dtype).unsqueeze(0)


def svf_threshold(p_emb: torch.Tensor, p_proj_emb: torch.Tensor, gamma: float = 10.0):
    """Sec 3.3 self-validating timestep threshold: t' = γ * sigmoid(1 - cos(p, p_proj))."""
    pf = p_emb.flatten().float()
    qf = p_proj_emb.flatten().float()
    cos = (pf @ qf) / (pf.norm() * qf.norm() + 1e-9)
    t_prime = gamma * torch.sigmoid(1.0 - cos)
    return float(t_prime.item())


# ---------------- main ----------------
def slugify(txt, maxlen=50):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", txt.strip())[:maxlen]


def load_prompts(f):
    f = Path(f)
    if f.suffix == ".csv":
        prompts, seeds = [], []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            pcol = next((c for c in ['adv_prompt','sensitive prompt','prompt','target_prompt','text']
                         if c in reader.fieldnames), None)
            scol = next((c for c in ['evaluation_seed','sd_seed','seed']
                         if c in reader.fieldnames), None)
            for row in reader:
                prompts.append(row[pcol].strip())
                seeds.append(int(row[scol]) if (scol and row.get(scol)) else None)
        return prompts, seeds
    return [l.strip() for l in open(f) if l.strip()], [None]


def build_concept_subspace(pipe, concept_terms, device, max_seq_len=256):
    """Encode each toxic term separately → concat into concept subspace [K_total, 4096]."""
    if not concept_terms:
        return None, None
    embeds_list = []
    pooled_list = []
    for term in concept_terms:
        with torch.no_grad():
            pe, _, ppe, _ = pipe.encode_prompt(
                prompt=term, prompt_2=term, prompt_3=term,
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                max_sequence_length=max_seq_len,
            )
        # pe: [1, L, 4096], ppe: [1, 2048]
        embeds_list.append(pe[0])           # [L, 4096]
        pooled_list.append(ppe[0])          # [2048]
    # Use mean over seq tokens as per-term embedding for concept subspace
    term_emb = torch.stack([e.mean(dim=0) for e in embeds_list], dim=0)  # [K, 4096]
    pooled_emb = torch.stack(pooled_list, dim=0)                          # [K, 2048]
    return term_emb, pooled_emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--concept", required=True, choices=list(CONCEPT_KEYWORDS.keys()))
    p.add_argument("--model_id", default=SD3_MODEL_ID)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--no_cpu_offload", action="store_true")
    # SAFREE flags (paper aligned)
    p.add_argument("--safree", action="store_true", help="Enable SAFREE token projection (Sec 3.2)")
    p.add_argument("--svf", "--self_validation_filter", action="store_true",
                   help="Enable self-validation filter (Sec 3.3)")
    p.add_argument("--lra", "--latent_re_attention", action="store_true",
                   help="(TODO) Enable Fourier re-attention (Sec 3.4) — currently NO-OP for SD3")
    p.add_argument("--sf_alpha", type=float, default=0.01)
    p.add_argument("--up_t", type=float, default=10.0, help="γ for SVF threshold")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Loading SD3 from {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    if args.no_cpu_offload:
        pipe = pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()

    if args.lra:
        print("[WARN] --lra not implemented for SD3 yet (TODO). Sec 3.4 Fourier re-attention skipped.")

    # Build concept subspace once
    if args.safree and args.concept != "none":
        terms = CONCEPT_KEYWORDS[args.concept]
        print(f"Building concept subspace from {len(terms)} toxic terms ...")
        concept_emb, concept_pooled = build_concept_subspace(pipe, terms, device, max_seq_len=256)
        print(f"  concept subspace: token-level={concept_emb.shape}  pooled={concept_pooled.shape}")
    else:
        concept_emb = concept_pooled = None

    prompts, seeds = load_prompts(args.prompts)
    if args.end is not None:
        prompts = prompts[args.start:args.end]; seeds = seeds[args.start:args.end]
    else:
        prompts = prompts[args.start:]; seeds = seeds[args.start:]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"SAFREE-SD3 v2: {len(prompts)} prompts, concept={args.concept}, "
          f"safree={args.safree}, svf={args.svf}, lra={args.lra}")

    gen = torch.Generator(device="cpu" if not args.no_cpu_offload else device)
    stats = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"SD3-SAFREE [{args.concept}]")):
        global_idx = args.start + i
        s = seeds[i] if seeds[i] is not None else args.seed + global_idx
        gen.manual_seed(s)

        # Standard prompt encoding
        with torch.no_grad():
            pe, neg_pe, ppe, neg_ppe = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None,    # NO negative_prompt — paper aligned
                max_sequence_length=256,
            )
        # pe: [1, L, 4096], ppe: [1, 2048]

        n_removed = -1
        used_safree = False
        if args.safree and concept_emb is not None:
            # Build pooled-input subspace I (Sec 3.2): use leave-one-out token-mean embeddings
            # Simpler: use full prompt mean as 1 vector + tokens themselves
            # Here we use the K toxic terms' pooled embeds + the prompt's own pooled as I
            with torch.no_grad():
                # Token-level projection
                input_subspace = torch.cat([
                    pe[0].mean(dim=0, keepdim=True),       # [1, 4096] — own
                    concept_emb,                            # [K, 4096]
                ], dim=0)                                   # [N, 4096]
                pe_proj, n_removed, dist = safree_project_embeds(
                    pe.to(device), concept_emb.to(device), input_subspace.to(device),
                    alpha=args.sf_alpha,
                )

                # Pooled projection (orthogonal only — no token mask)
                ppe_proj = safree_project_pooled(
                    ppe.to(device), concept_pooled.to(device),
                )

                # SVF: gate whole-prompt projection by cosine threshold
                if args.svf:
                    t_prime = svf_threshold(pe[0], pe_proj[0], gamma=args.up_t)
                    # If t_prime is small (cos near 1, prompt benign), use original prompt
                    # Threshold heuristic: if t_prime < 0.5, prompt is very safe → skip projection
                    if t_prime < 0.5:
                        pe_use = pe; ppe_use = ppe
                    else:
                        pe_use = pe_proj; ppe_use = ppe_proj
                    print(f"  prompt {global_idx}: removed={n_removed}/{pe.shape[1]} t'={t_prime:.3f} -> "
                          f"{'SAFREE' if pe_use is pe_proj else 'PASS'}")
                else:
                    pe_use = pe_proj; ppe_use = ppe_proj
                used_safree = (pe_use is pe_proj)
        else:
            pe_use = pe; ppe_use = ppe

        with torch.no_grad():
            img = pipe(
                prompt_embeds=pe_use,
                pooled_prompt_embeds=ppe_use,
                negative_prompt_embeds=neg_pe,
                negative_pooled_prompt_embeds=neg_ppe,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg_scale,
                height=args.resolution, width=args.resolution,
                generator=gen,
            ).images[0]

        name = slugify(prompt)
        img.save(str(outdir / f"{global_idx:04d}_00_{name}.png"))
        stats.append({"idx": global_idx, "n_removed": n_removed, "used_safree": used_safree})

    json.dump({"method": "safree_sd3_v2", "concept": args.concept,
               "args": vars(args), "stats": stats},
              open(outdir / "stats.json", "w"), indent=2)
    print(f"Done. {len(prompts)} images saved to {outdir}")


if __name__ == "__main__":
    main()
