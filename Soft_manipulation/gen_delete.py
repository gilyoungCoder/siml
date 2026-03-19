# sd14_from_file_softdelete_prob.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, Attention


# =========================
# Utils & IO
# =========================
def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # drop blanks and comments
    return [ln for ln in lines if ln and not ln.lstrip().startswith("#")]


def read_terms(path: str) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def sanitize_filename(text: str, maxlen: int = 80):
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-.,+=@()\[\]]+", "_", text)
    return (text[:maxlen]).rstrip("_")


def banner(title: str):
    bar = "=" * 70
    print(f"{bar}\n= {title}\n{bar}")


# =========================
# Text encode helpers
# =========================
@torch.no_grad()
def encode_texts(pipe: StableDiffusionPipeline, texts: List[str]):
    tok = pipe.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    out.input_ids = tok.input_ids
    out.attention_mask = tok.attention_mask
    return out


def pick_hidden(out, layer_idx: int):
    # layer_idx=-1 -> last_hidden_state; else pick from hidden_states
    if hasattr(out, "hidden_states") and out.hidden_states is not None and layer_idx is not None and layer_idx != -1:
        return out.hidden_states[layer_idx]
    return out.last_hidden_state


def build_content_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, include_special: bool):
    mask = attention_mask.bool()
    if not include_special:
        for sid in [
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        ]:
            if sid is not None:
                mask = mask & (input_ids != sid)
    return mask


@torch.no_grad()
def build_harm_vec(
    pipe: StableDiffusionPipeline,
    terms: List[str],
    layer_idx: int = -1,
    include_special: bool = False,
) -> Optional[torch.Tensor]:
    """masked mean over content tokens across all harm terms, L2-normalized"""
    if not terms:
        return None
    out = encode_texts(pipe, terms)
    H = pick_hidden(out, layer_idx)  # (N,L,D)
    H = F.normalize(H, dim=-1)
    mask = build_content_mask(pipe.tokenizer, out.input_ids, out.attention_mask, include_special)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    v_per = (H * mask.unsqueeze(-1)).sum(dim=1) / denom  # (N,D)
    v = F.normalize(v_per.mean(dim=0), dim=-1)  # (D,)
    return v


@torch.no_grad()
def prompt_token_cosine_stats(
    pipe: StableDiffusionPipeline,
    prompt: str,
    harm_vec: torch.Tensor,
    layer_idx: int = -1,
    include_special: bool = False,
    topk: int = 8,
) -> Tuple[List[Tuple[str, float]], float, float, float]:
    """
    Return:
      - topk tokens by cosine (token_str, cos)
      - cos_max, cos_mean(>0), (caller computes ≥tau%)
    """
    out = encode_texts(pipe, [prompt])
    H = pick_hidden(out, layer_idx)[0]  # (L,D)
    Hn = F.normalize(H, dim=-1)
    mask = build_content_mask(pipe.tokenizer, out.input_ids[0], out.attention_mask[0], include_special)
    toks = pipe.tokenizer.convert_ids_to_tokens(out.input_ids[0].tolist())

    harm = F.normalize(harm_vec.detach(), dim=0).to(device=H.device, dtype=H.dtype)
    cos_all = (Hn @ harm)  # (L,)

    cos_masked = cos_all[mask]
    toks_masked = [t for t, m in zip(toks, mask.tolist()) if m]
    if cos_masked.numel() == 0:
        return [], 0.0, 0.0, 0.0

    vals, idxs = torch.topk(cos_masked, k=min(topk, cos_masked.shape[0]))
    top = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        token = toks_masked[i].replace("Ġ", " ")
        top.append((token, float(v)))

    cos_max = float(cos_masked.max().item())
    cos_pos = cos_masked[cos_masked > 0]
    cos_mean_pos = float(cos_pos.mean().item()) if cos_pos.numel() > 0 else 0.0
    return top, cos_max, cos_mean_pos, float(cos_masked.numel())


# =========================
# Soft Delete Processor (probability redistribution)
# =========================
class SoftDeleteCfg:
    __slots__ = ("enable", "tau", "gamma")

    def __init__(self, enable=True, tau=0.12, gamma=0.45):
        self.enable = bool(enable)
        self.tau = float(tau)
        self.gamma = float(gamma)


class SoftDeleteAttnProcessor(AttnProcessor2_0):
    """
    Cross-Attn Soft Delete (harm-only), **on probabilities**:
      1) scores -> softmax -> attn_probs
      2) for key-columns with cos(harm) >= tau:
            factor = clamp(1 - gamma * relu(cos - tau), min=0)
            attn_probs[..., k] *= factor
      3) row-wise renormalize over K (keys)

    Exempt BOS/SOT column if a mask is provided.
    Runtime stats are stored for step-wise debug.
    """

    def __init__(self, harm_vec: Optional[torch.Tensor], cfg: SoftDeleteCfg):
        super().__init__()
        self.cfg = cfg
        self._harm = F.normalize(harm_vec.detach().float(), dim=-1).cpu() if harm_vec is not None else None
        self._soft_exempt_mask = None  # (B,K) True means exempt (no delete)

        # debug stats
        self.last_tau = float(cfg.tau)
        self.last_gamma = float(cfg.gamma)
        self.last_harmful_pct = 0.0
        self.last_w_mean = 0.0
        self.last_w_max = 0.0
        self.last_shapes = (0, 0, 0, 0)  # (B, H_heads, Q, K)
        self.last_applied = False
        self.last_avg_row_scale = 1.0  # 평균 reweight scale (정보용)

    def set_gamma(self, g: float):
        self.cfg.gamma = float(g)
        self.last_gamma = float(g)

    def set_harm_vec(self, v: Optional[torch.Tensor]):
        self._harm = F.normalize(v.detach().float(), dim=-1).cpu() if v is not None else None

    def set_soft_exempt_mask(self, m: Optional[torch.Tensor]):
        self._soft_exempt_mask = m.bool().detach().cpu() if m is not None else None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb=None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:

        dev = hidden_states.device
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q, K)

        if attention_mask is not None:
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        # --- softmax to get probabilities ---
        attn_probs = torch.softmax(scores, dim=-1)

        # --- probability redistribution (our change) ---
        self.last_applied = False
        self.last_avg_row_scale = 1.0
        if is_cross and self.cfg.enable and (self._harm is not None) and (self.cfg.gamma > 0):
            txt_n = F.normalize(encoder_hidden_states, dim=-1)  # (B,K,D)
            harm = self._harm.to(device=dev, dtype=txt_n.dtype)
            cos_harm = torch.einsum("bkd,d->bk", txt_n, harm)  # (B,K)
            cond = cos_harm >= self.cfg.tau  # (B,K)

            if self._soft_exempt_mask is not None:
                exc = self._soft_exempt_mask.to(dev)
                if exc.shape[0] == 1:
                    exc = exc.expand(cond.shape[0], -1)
                cond = cond & (~exc)

            if cond.any():
                B, K = cond.shape
                Q = attn_probs.shape[1]
                Hh = attn_probs.shape[0] // B
                self.last_shapes = (int(B), int(Hh), int(Q), int(K))

                w = (cos_harm - self.cfg.tau).clamp_min(0.0) * cond.float()  # (B,K)
                self.last_harmful_pct = float(cond.float().mean().item() * 100.0)
                pos = w[w > 0]
                self.last_w_mean = float(pos.mean().item()) if pos.numel() > 0 else 0.0
                self.last_w_max = float(w.max().item()) if pos.numel() > 0 else 0.0

                # multiplicative factor in [0,1]; stronger cos -> smaller factor
                factor = (1.0 - self.cfg.gamma * w).clamp(min=0.0)  # (B,K)
                factor = (
                    factor[:, None, :]  # (B,1,K)
                    .expand(B, Q, K)
                    .repeat_interleave(Hh, dim=0)  # (B*Hh,Q,K)
                    .to(attn_probs.dtype)
                )

                # apply factor and renormalize row-wise over K
                attn_probs = attn_probs * factor
                row_sum = attn_probs.sum(dim=-1, keepdim=True)  # (B*Hh,Q,1)
                eps = 1e-12
                self.last_avg_row_scale = float((row_sum.mean().clamp(min=eps)).item())
                attn_probs = attn_probs / (row_sum.clamp(min=eps))
                self.last_applied = True

        # dropout (kept same semantics as diffusers)
        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout, (int, float)) else 0.0
            attn_probs = torch.dropout(attn_probs, p=p, train=False)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def gamma_schedule_linear(step: int, num_steps: int, g_start: float, g_end: float) -> float:
    if num_steps <= 1:
        return g_end
    t = step / (num_steps - 1)
    return g_start * (1.0 - t) + g_end * t


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Stable Diffusion v1-4: generate images from a prompt file (+ Soft Delete)")

    # model/IO
    p.add_argument("--model", default="CompVis/stable-diffusion-v1-4", help="HF model id or local path")
    p.add_argument("--prompt_file", required=True, help="Path to a text file: one prompt per line")
    p.add_argument("--output_dir", default="outputs/sd14", help="Output directory")

    # generation
    p.add_argument("--negative_prompt", default="nudity",
                   help='Negative prompt (default: "nudity"). Use "" to disable.')
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234, help="Base seed (each prompt offsets by its index)")
    p.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto",
                   help="Computation dtype (auto=fp16 on CUDA else fp32)")
    p.add_argument("--max_prompts", type=int, default=0, help="If >0, limit number of prompts")

    # soft delete
    p.add_argument("--enable_soft_delete", action="store_true", help="Enable cross-attn soft delete (probability mode)")
    p.add_argument("--harm_texts", type=str, default=None, help="Path to harm terms file (one per line)")
    p.add_argument("--tau", type=float, default=0.12, help="Cosine threshold for soft delete")
    p.add_argument("--gamma_start", type=float, default=0.0, help="Start gamma")
    p.add_argument("--gamma_end", type=float, default=0.0, help="End gamma")
    p.add_argument("--layer_index", type=int, default=-1, help="-1 uses last hidden state; else specific hidden layer")
    p.add_argument("--include_special_tokens", action="store_true", help="Include BOS/EOS/PAD into content averaging")
    p.add_argument("--print_steps", type=int, default=5, help="Step logging interval (0 disables)")
    p.add_argument("--debug", action="store_true", help="Verbose logs")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(args.prompt_file)
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    banner("Stable Diffusion v1-4: batch generation (+ Soft Delete prob-redistribution)")
    print(f"Model:              {args.model}")
    print(f"Prompts file:       {args.prompt_file}  (N={len(prompts)})")
    print(f"Output dir:         {outdir.resolve()}")
    print(f"Device / dtype:     {device} / {str(torch_dtype).replace('torch.', '')}")
    print(f"Steps / Scale:      {args.num_inference_steps} / {args.guidance_scale}")
    print(f"Size (HxW):         {args.height}x{args.width}")
    print(f"Negative prompt:    {repr(args.negative_prompt)}")
    print(f"Base seed:          {args.seed}")
    if args.enable_soft_delete:
        print(f"SoftDelete:         ON (prob) | tau={args.tau:.3f}, gamma={args.gamma_start:.3f}->{args.gamma_end:.3f}, layer={args.layer_index}")
        print(f"Harm terms file:    {args.harm_texts}")
    else:
        print("SoftDelete:         OFF")

    # ---- load pipeline (safety checker OFF) ----
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing("max")

    # ---- build harm vec & install processor ----
    sd_proc = None
    harm_vec = None
    if args.enable_soft_delete:
        harm_terms = read_terms(args.harm_texts) if args.harm_texts else []
        if len(harm_terms) == 0:
            print("[WARN] Soft delete enabled but harm terms file is empty or missing → disabling soft delete.")
        else:
            harm_vec = build_harm_vec(
                pipe, harm_terms, layer_idx=args.layer_index, include_special=args.include_special_tokens
            )
            if harm_vec is None:
                print("[WARN] Failed to build harm vector → disabling soft delete.")
            else:
                sd_proc = SoftDeleteAttnProcessor(harm_vec, cfg=SoftDeleteCfg(True, args.tau, args.gamma_start))
                pipe.unet.set_attn_processor(sd_proc)

    # ---- step callback (gamma schedule + per-step stats) ----
    def on_step_end(diffusion_pipeline, step: int, timestep: int, callback_kwargs: dict):
        if sd_proc is not None:
            g = gamma_schedule_linear(step, args.num_inference_steps, args.gamma_start, args.gamma_end)
            sd_proc.set_gamma(g)
        if args.print_steps > 0 and (step % args.print_steps == 0):
            if sd_proc is not None:
                B, Hh, Q, K = sd_proc.last_shapes
                print(
                    f"[step {step:>3}] t={int(timestep)} | gamma={sd_proc.last_gamma:.3f} | "
                    f"applied={sd_proc.last_applied} | harmful%={sd_proc.last_harmful_pct:.2f} | "
                    f"w_mean/max={sd_proc.last_w_mean:.4f}/{sd_proc.last_w_max:.4f} | "
                    f"avg_row_scale={sd_proc.last_avg_row_scale:.4f} | "
                    f"shapes(B,H,Q,K)={B},{Hh},{Q},{K}"
                )
            else:
                print(f"[step {step:>3}] t={int(timestep)} | (soft-delete OFF)")
        return callback_kwargs

    cb = on_step_end if args.print_steps > 0 else None
    cb_inputs = ["latents"] if cb is not None else None

    # ---- generation loop ----
    for idx, prompt in enumerate(prompts, start=1):
        g = torch.Generator(device=device).manual_seed(args.seed + idx)
        neg = None if args.negative_prompt == "" else args.negative_prompt
        print(f"\n[{idx:03d}/{len(prompts)}] prompt={prompt!r} | negative={neg!r}")

        # (DBG) per-prompt cosine stats vs harm vector (conditional encoder side)
        if args.enable_soft_delete and harm_vec is not None:
            top, cos_max, cos_mean_pos, _ = prompt_token_cosine_stats(
                pipe,
                prompt,
                harm_vec,
                layer_idx=args.layer_index,
                include_special=args.include_special_tokens,
                topk=8,
            )
            # recompute pct_over_tau with same mask:
            out = encode_texts(pipe, [prompt])
            H = pick_hidden(out, args.layer_index)[0]
            Hn = F.normalize(H, dim=-1)
            harm = F.normalize(harm_vec, dim=0).to(H.device, H.dtype)
            mask = build_content_mask(pipe.tokenizer, out.input_ids[0], out.attention_mask[0], args.include_special_tokens)
            cos = (Hn @ harm)[mask]
            pct_over_tau = float(((cos >= args.tau).float().mean().item() * 100.0)) if cos.numel() > 0 else 0.0

            if args.debug:
                print("  [cosine] top tokens:", ", ".join([f"{t.strip()}:{v:.3f}" for t, v in top]))
                print(f"  [cosine] max={cos_max:.3f} | mean_pos={cos_mean_pos:.3f} | ≥tau%={pct_over_tau:.2f}")

            # SOT-exempt mask (protect BOS column)
            ids = out.input_ids[0]
            att = out.attention_mask[0].bool()
            bos = getattr(pipe.tokenizer, "bos_token_id", None)
            sot_mask = torch.zeros_like(ids, dtype=torch.bool)
            if bos is not None:
                sot_mask = att & (ids == bos)
            else:
                first = int(att.nonzero(as_tuple=False)[0].item())
                sot_mask[first] = True
            if sd_proc is not None:
                sd_proc.set_soft_exempt_mask(sot_mask.unsqueeze(0))

        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=g,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=cb_inputs,
        )

        image = result.images[0]
        base = sanitize_filename(prompt) or f"prompt_{idx:03d}"
        save_path = outdir / f"{idx:03d}_{base}.png"
        image.save(save_path)
        print(f" -> saved: {save_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
