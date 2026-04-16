"""
FLUX.2-klein Spatial Probe — simplified WHERE component.

Unlike SD v1.4 cross-attention probing (which projects image features through
to_k of each attn2 layer and attends to target text tokens), FLUX uses joint
attention which makes that approach harder. We instead:

  1. Forward-hook ONE transformer block in the DiT and capture its IMAGE-token
     output  (shape [B, seq_img, C_inner])
  2. Reshape to a 2D spatial grid (seq_img = (H/2) * (W/2) in packed latent space)
  3. Compute cosine similarity between each spatial feature vector and a
     projected target embedding (CLIP-style: reduce text over tokens, then
     project through a small linear map to match feature dim; here we simply
     average and L2-normalise since hidden dims can differ — we handle that by
     projecting the larger one with a random but fixed seeded linear layer OR,
     in the minimal path, by using the transformer's own text-token stream
     captured alongside the image tokens from the same block output).
  4. → 2D mask [H/2, W/2]; upsample to latent resolution [H, W].

The third step above is key: joint attention blocks return a single tensor
[B, seq_txt + seq_img, C_inner] whose image and text halves live in the SAME
feature space.  So the cleanest similarity path is:

     feat_img  = block_out[:, -seq_img:, :]     # [B, seq_img, C]
     feat_txt  = block_out[:, :-seq_img, :]     # [B, seq_txt, C]
     t_vec     = feat_txt.mean(dim=1)           # [B, C]
     mask_flat = cos(feat_img, t_vec)           # [B, seq_img]

This gives a per-spatial-position score of "how similar is this patch to the
conditional text stream right now" which is exactly what the text probe does
in SD v1.4 — only now computed from the SAME forward pass, no extra calls.

For the TEXT-PROBE variant we further gate by comparing each spatial position
against a fixed target concept (nudity/etc.) via the target-conditioned
forward that generate_flux2klein_v1.py already runs (et pass).  The probe just
stores block output on EACH of {prompt pass, target pass} and we contrast
their image-feature cosine-sims at the end of the step.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class FluxSpatialProbe:
    """Hook ONE FLUX transformer block and capture its image-token output.

    Supports capturing multiple passes within a denoising step (prompt pass,
    target pass) by switching `self.tag` before each forward.
    """

    def __init__(self, block, seq_img_len: int):
        """
        Args:
            block: a transformer block module (e.g. transformer.transformer_blocks[10])
            seq_img_len: number of image tokens in the packed latent sequence
                         ( = (H/2) * (W/2) ).  Used to slice the image half.
        """
        self.block = block
        self.seq_img_len = seq_img_len
        self.active = False
        self.tag: str = "prompt"
        self.captures: dict = {}
        self._handle = block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.active:
            return
        # FLUX blocks return either a tensor or a tuple; take first tensor
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if not torch.is_tensor(out) or out.dim() != 3:
            return
        # FLUX single-stream blocks output [B, seq_txt + seq_img, C].
        # Image tokens are the LAST seq_img_len positions.
        S = out.shape[1]
        if S == self.seq_img_len:
            img = out.detach()
        elif S > self.seq_img_len:
            img = out[:, -self.seq_img_len:, :].detach()
        else:
            return
        self.captures[self.tag] = img

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self):
        self.captures.clear()


def compute_flux_spatial_mask(
    feat: torch.Tensor,
    target_vec: Optional[torch.Tensor] = None,
    target_feat: Optional[torch.Tensor] = None,
    threshold: float = 0.1,
    mode: str = "text",
) -> torch.Tensor:
    """Compute a 2D mask from block-output image features.

    Args:
        feat: [B, seq_img, C] image-token features from the probe (prompt pass)
        target_vec: [B, C] or [C] — single vector to compare against (text mode
            using pooled text tokens).  Ignored if `target_feat` is given.
        target_feat: [B, seq_img, C] — image-token features from the target
            pass; enables contrast (prompt-vs-target) mask.
        threshold: values below this (after normalisation to [0,1]) are set
            to this floor so the mask never fully cancels guidance.
        mode: "text" (use target_vec) | "contrast" (use target_feat)

    Returns:
        mask: [B, H2, W2] float in [threshold, 1].
    """
    B, S, C = feat.shape
    side = int(round(S ** 0.5))
    assert side * side == S, f"seq_img={S} not a perfect square"

    feat_n = F.normalize(feat.float(), dim=-1)

    if mode == "contrast" and target_feat is not None:
        tf_n = F.normalize(target_feat.float(), dim=-1)
        # Element-wise similarity: prompt patch vs target patch at same position
        sim = (feat_n * tf_n).sum(dim=-1)  # [B, S]
    else:
        if target_vec is None:
            # Fallback: self-energy — norm of each spatial feature
            sim = feat.float().norm(dim=-1)
        else:
            if target_vec.dim() == 1:
                target_vec = target_vec.unsqueeze(0).expand(B, -1)
            tv = F.normalize(target_vec.float(), dim=-1).unsqueeze(1)  # [B,1,C]
            sim = (feat_n * tv).sum(dim=-1)  # [B, S]

    # Normalise per-sample to [0, 1]
    mask = sim.view(B, side, side)
    vmin = mask.reshape(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
    vmax = mask.reshape(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
    mask = (mask - vmin) / (vmax - vmin + 1e-8)
    mask = mask.clamp(min=threshold, max=1.0)
    return mask  # [B, side, side]


def upsample_mask_to_latent(mask: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    """Bilinear upsample [B, h, w] -> [B, 1, H, W]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    m = mask.unsqueeze(1).float()
    m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
    return m  # [B, 1, H, W]


def mask_to_packed_seq(mask_2d: torch.Tensor, seq_img_len: int) -> torch.Tensor:
    """Downsample a [B,1,H,W] or [B,h,w] mask to a packed sequence [B, seq, 1].

    FLUX.2-klein latents are packed via a 2x2 patchify: seq = (H/2)*(W/2).
    For guidance we want a scalar weight per image token.
    """
    if mask_2d.dim() == 2:
        mask_2d = mask_2d.unsqueeze(0)
    if mask_2d.dim() == 3:
        mask_2d = mask_2d.unsqueeze(1)
    B = mask_2d.shape[0]
    side = int(round(seq_img_len ** 0.5))
    m = F.interpolate(mask_2d.float(), size=(side, side),
                      mode="bilinear", align_corners=False)
    m = m.reshape(B, side * side, 1)
    return m  # [B, seq_img_len, 1]
