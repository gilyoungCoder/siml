"""
Joint-Attention Probing for SD3's MMDiT Architecture.

SD3's JointAttnProcessor2_0 concatenates [image_tokens, text_tokens] for
joint self-attention. We extract the image→text sub-matrix as a
cross-attention-like spatial map for concept localization.

Attention matrix layout (concatenation order: [image, text]):
  Q = [Q_img; Q_txt],  K = [K_img; K_txt]
  attn = softmax(Q @ K^T / sqrt(d))

  Cross-attention-like = attn[0:num_img, num_img:num_img+num_txt]
    = how each image patch attends to each text token

Key difference from SD1.4 UNet:
  - SD1.4: separate cross-attention layers (attn2) with separate Q(img), K(text)
  - SD3: joint attention with concatenated [img, txt] tokens
  - SD1.4: uses attn.to_q, attn.to_k for image, text comes via encoder_hidden_states
  - SD3: uses attn.to_q/k/v for image, attn.add_q_proj/k_proj/v_proj for text
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F


class SD3AttentionProbeStore:
    """Stores probe attention maps extracted from SD3 joint attention blocks."""

    def __init__(self):
        self.probe_maps: Dict[str, torch.Tensor] = {}
        self.active: bool = False

    def store_probe(self, block_name: str, attn_map: torch.Tensor):
        self.probe_maps[block_name] = attn_map.detach()

    def reset(self):
        self.probe_maps.clear()

    def get_maps(self) -> Dict[str, torch.Tensor]:
        return self.probe_maps


class JointAttentionProbeProcessor:
    """
    Custom processor for SD3 joint attention that:
    1. Performs normal joint attention (output unchanged)
    2. Extracts image→text attention sub-matrix
    3. Stores it in SD3AttentionProbeStore

    Replaces JointAttnProcessor2_0 on selected blocks.
    """

    def __init__(self, store: SD3AttentionProbeStore, block_name: str):
        self.store = store
        self.block_name = block_name

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        num_img_tokens = hidden_states.shape[1]

        # Image projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Text projections (context)
        num_txt_tokens = 0
        if encoder_hidden_states is not None:
            num_txt_tokens = encoder_hidden_states.shape[1]

            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            # Concatenate: [image, text]
            query = torch.cat([query, enc_q], dim=2)
            key = torch.cat([key, enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        if self.store.active and num_txt_tokens > 0:
            # Compute attention explicitly to capture weights
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = attn_weights.softmax(dim=-1)
            hidden_out = torch.matmul(attn_probs, value)

            # Extract image→text sub-matrix
            # attn_probs: [B, H, num_img+num_txt, num_img+num_txt]
            # We want: [H, num_img, num_txt] for the conditional batch
            b_idx = min(1, batch_size - 1)  # Use conditional (idx=1 in CFG)
            cross_attn = attn_probs[b_idx, :, :num_img_tokens, num_img_tokens:]
            # Shape: [H, num_img, num_txt]
            self.store.store_probe(self.block_name, cross_attn)
        else:
            # Use SDPA for speed when not probing
            hidden_out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

        # Reshape and split
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_out = hidden_out.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states = hidden_out[:, :num_img_tokens]
            encoder_hidden_states = hidden_out[:, num_img_tokens:]

            # Output projections
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            hidden_states = attn.to_out[0](hidden_out)
            hidden_states = attn.to_out[1](hidden_out)
            return hidden_states


def register_sd3_attention_probe(
    transformer,
    store: SD3AttentionProbeStore,
    target_blocks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Register JointAttentionProbeProcessor on target transformer blocks.

    Args:
        transformer: SD3Transformer2DModel
        store: SD3AttentionProbeStore
        target_blocks: Block indices to hook (None = middle third)

    Returns:
        {block_name: original_processor} for restoration
    """
    num_blocks = len(transformer.transformer_blocks)

    if target_blocks is None:
        # Middle third of blocks — most informative for spatial attention
        start = num_blocks // 3
        end = 2 * num_blocks // 3
        target_blocks = list(range(start, end))

    original_processors = {}
    hooked = 0

    for block_idx in target_blocks:
        if block_idx >= num_blocks:
            continue

        block = transformer.transformer_blocks[block_idx]
        block_name = f"block_{block_idx}"

        # Save original
        original_processors[block_name] = block.attn.processor

        # Replace with probe processor
        block.attn.processor = JointAttentionProbeProcessor(
            store=store,
            block_name=block_name,
        )
        hooked += 1

    print(f"  [SD3 probe] Hooked {hooked} joint attention blocks: {target_blocks}")
    return original_processors


def restore_sd3_processors(transformer, original_processors: Dict[str, Any]):
    """Restore original attention processors."""
    for block_name, proc in original_processors.items():
        block_idx = int(block_name.split("_")[1])
        if block_idx < len(transformer.transformer_blocks):
            transformer.transformer_blocks[block_idx].attn.processor = proc


def compute_sd3_spatial_mask(
    store: SD3AttentionProbeStore,
    token_indices: Optional[List[int]] = None,
    latent_h: int = 64,
    latent_w: int = 64,
) -> torch.Tensor:
    """
    Aggregate SD3 joint attention probe maps into a spatial mask.

    SD3-medium at 1024x1024: latent 128x128 with patch_size=2 → 64x64 = 4096 image tokens.

    Args:
        store: SD3AttentionProbeStore
        token_indices: Text token indices to focus on (None = all)
        latent_h, latent_w: Spatial dims for reshaping image tokens to 2D

    Returns:
        spatial_mask: [H, W] normalized in [0, 1]
    """
    maps = store.get_maps()
    if not maps:
        return torch.zeros(latent_h, latent_w)

    all_maps = []
    for block_name, attn_map in maps.items():
        # attn_map: [H, num_img_tokens, num_txt_tokens]

        # Average across heads
        avg_map = attn_map.mean(dim=0)  # [num_img, num_txt]

        # Select target text tokens
        if token_indices is not None and len(token_indices) > 0:
            token_map = avg_map[:, token_indices]
        else:
            token_map = avg_map

        # Max across text tokens (union)
        spatial_map = token_map.max(dim=-1)[0]  # [num_img]

        # Reshape to 2D
        num_patches = spatial_map.shape[0]
        ph = int(num_patches ** 0.5)
        pw = num_patches // ph
        spatial_2d = spatial_map.view(1, 1, ph, pw)

        # Interpolate to target resolution
        if ph != latent_h or pw != latent_w:
            spatial_2d = F.interpolate(
                spatial_2d.float(), size=(latent_h, latent_w),
                mode="bilinear", align_corners=False,
            )

        all_maps.append(spatial_2d)

    if not all_maps:
        return torch.zeros(latent_h, latent_w)

    combined = torch.stack(all_maps).mean(0)

    # Normalize to [0, 1]
    flat = combined.reshape(-1)
    vmin, vmax = flat.min(), flat.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)

    return combined.squeeze(0).squeeze(0)
