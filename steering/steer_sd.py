#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import os

# ============================================================
# 1. 환경 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ============================================================
# 2. Helper 함수
# ============================================================
def orth(A: torch.Tensor) -> torch.Tensor:
    if A.numel() == 0:
        return A
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q

def build_pure_spaces(Eh: torch.Tensor, Es: torch.Tensor, tau: float = 0.9):
    Uh = orth(Eh.T)   # d×kh
    Us = orth(Es.T)   # d×ks
    M = Uh.T @ Us
    A, S, B = torch.linalg.svd(M)
    mask = (S >= tau)
    Uh_cap = Uh @ A[:, mask] if mask.any() else Uh.new_zeros(Uh.size(0), 0)
    Us_cap = Us @ B[:, mask] if mask.any() else Us.new_zeros(Us.size(0), 0)
    Ucap   = orth(torch.cat([Uh_cap, Us_cap], dim=1)) if mask.any() else Uh.new_zeros(Uh.size(0), 0)
    Pcap   = Ucap @ Ucap.T if Ucap.numel() else torch.zeros((Uh.size(0), Uh.size(0)), device=Uh.device)
    Uh_pure = orth((torch.eye(Uh.size(0), device=Uh.device) - Pcap) @ Uh)
    Us_pure = orth((torch.eye(Us.size(0), device=Us.device) - Pcap) @ Us)
    Ph = Uh_pure @ Uh_pure.T if Uh_pure.numel() else torch.zeros_like(Pcap)
    Ps = Us_pure @ Us_pure.T if Us_pure.numel() else torch.zeros_like(Pcap)
    return Ph, Ps, Pcap

def steer_tokens(E: torch.Tensor,
                 Ph: torch.Tensor,
                 Ps: torch.Tensor,
                 gamma0: float = 1.0,
                 lambda0: float = 0.5,
                 p: float = 1.0,
                 q: float = 1.0,
                 eps: float = 1e-6,
                 log_prefix: str = "",
                 log_plot: bool = False,
                 outdir: str = "outputs") -> torch.Tensor:
    B, T, d = E.shape
    E_flat = E.reshape(-1, d)

    eh = torch.linalg.vector_norm(E_flat @ Ph, dim=-1)
    es = torch.linalg.vector_norm(E_flat @ Ps, dim=-1)
    r  = eh / (eh + es + eps)

    gamma = gamma0 * (r ** p)
    lam   = lambda0 * ((1 - r) ** q)

    # 로깅
    for i in range(min(10, len(r))):
        print(f"{log_prefix} token {i}: eh={eh[i]:.4f}, es={es[i]:.4f}, γ={gamma[i]:.3f}, λ={lam[i]:.3f}")

    # 그래프 저장
    if log_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(eh.cpu(), label="harm energy (eh)")
        plt.plot(es.cpu(), label="safe energy (es)")
        plt.plot(gamma.cpu(), label="gamma (harm suppression)")
        plt.plot(lam.cpu(), label="lambda (safe boost)")
        plt.xlabel("Token index")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Token-wise Steering Strength")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"steering_strength.png"))
        plt.close()

    E_out = E_flat - (gamma[:, None] * (E_flat @ Ph)) + (lam[:, None] * (E_flat @ Ps))
    return E_out.reshape(B, T, d)

# ============================================================
# 3. Harm / Safe Prompt 세트 정의
# ============================================================
harm_prompts = [
    "nude", "naked", "undressed", "bare body", "exposed body",
    "without clothes", "nude person", "naked woman", "naked man"
]

safe_prompts = [
    "clothed", "covered", "fully clothed", "dressed", "wearing clothes",
    "with outfit", "covered body", "wearing shirt", "in clothing", "modestly dressed"
]

# ============================================================
# 4. CLIP 임베딩 준비
# ============================================================
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

def encode_prompts(prompts):
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_encoder(**tokens)
    return outputs.last_hidden_state

Eh = encode_prompts(harm_prompts).reshape(-1, text_encoder.config.hidden_size)
Es = encode_prompts(safe_prompts).reshape(-1, text_encoder.config.hidden_size)

Ph, Ps, Pcap = build_pure_spaces(Eh, Es, tau=0.9)

# ============================================================
# 5. Stable Diffusion Pipeline Hook
# ============================================================
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
orig_forward = pipe.text_encoder.forward

def steered_forward(*args, **kwargs):
    outputs = orig_forward(*args, **kwargs)
    last_hidden = outputs.last_hidden_state
    steered = steer_tokens(last_hidden, Ph, Ps,
                           gamma0=1.0, lambda0=0.5, p=1.0, q=1.0,
                           log_prefix="[STEER]", log_plot=True, outdir="outputs")
    return outputs.__class__(last_hidden_state=steered, pooler_output=outputs.pooler_output)

# ============================================================
# 6. 실행 (원본 vs Steered)
# ============================================================
if __name__ == "__main__":
    prompt = "a country nude body, photorealistic"
    os.makedirs("outputs", exist_ok=True)

    # 원본
    pipe.text_encoder.forward = orig_forward
    img_orig = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    img_orig.save("outputs/original.png")
    print("Saved outputs/original.png")

    # Steered
    pipe.text_encoder.forward = steered_forward
    img_steered = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    img_steered.save("outputs/steered.png")
    print("Saved outputs/steered.png")
