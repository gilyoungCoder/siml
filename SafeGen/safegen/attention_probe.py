"""
Cross-Attention Probing for Spatial Unsafe Concept Detection.

During the standard UNet forward pass, probes each cross-attention layer with
pre-cached target concept keys to determine WHERE unsafe content is being
generated. Zero extra UNet calls — just one matmul per attention layer.

Architecture:
    - AttentionProbeStore: collects probe attention maps per denoising step
    - ProbeCrossAttnProcessor: single-probe (text OR image)
    - DualProbeCrossAttnProcessor: dual-probe (text AND image) in one pass
    - precompute_target_keys: one-time K_target computation per layer
    - compute_attention_spatial_mask: aggregates probe maps into spatial mask

Usage:
    store = AttentionProbeStore()
    target_keys = precompute_target_keys(unet, target_embeds, [16, 32])
    orig_procs = register_attention_probe(unet, store, target_keys, [16, 32])

    # During denoising:
    store.active = True
    store.reset()
    eps = unet(...)  # probe maps captured automatically
    store.active = False

    mask = compute_attention_spatial_mask(store, token_indices)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


# SD1.4 UNet block name -> spatial resolution
BLOCK_RESOLUTION = {
    "down_blocks.0": 64,
    "down_blocks.1": 32,
    "down_blocks.2": 16,
    "mid_block": 8,
    "up_blocks.1": 16,
    "up_blocks.2": 32,
    "up_blocks.3": 64,
}


def _get_resolution(layer_name: str) -> int:
    """Infer spatial resolution from attention layer name."""
    for prefix, res in BLOCK_RESOLUTION.items():
        if layer_name.startswith(prefix):
            return res
    return 0


class AttentionProbeStore:
    """Stores probe attention maps from cross-attention processors."""

    def __init__(self):
        self.probe_maps: Dict[str, torch.Tensor] = {}
        self.active: bool = False

    def store_probe(self, layer_name: str, attn_probs: torch.Tensor):
        """Store probe attention. Shape: (H, spatial_seq, num_tokens)"""
        self.probe_maps[layer_name] = attn_probs.detach()

    def reset(self):
        self.probe_maps.clear()

    def get_maps(self) -> Dict[str, torch.Tensor]:
        return self.probe_maps


class ProbeCrossAttnProcessor:
    """
    Custom attention processor that performs normal cross-attention AND computes
    probe attention with target concept keys. Only probes the conditional half
    of the CFG batch (index 1 when B=2).
    """

    def __init__(
        self,
        store: AttentionProbeStore,
        layer_name: str,
        target_key: torch.Tensor,
    ):
        self.store = store
        self.layer_name = layer_name
        self.target_key = target_key

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        enc_states = encoder_hidden_states if is_cross else hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(enc_states)
        value = attn.to_v(enc_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if is_cross and self.store.active:
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = attn_weights.softmax(dim=-1)
            hidden_out = torch.matmul(attn_probs, value)

            # Probe: compute attention with target concept keys
            q_cond = query[1:2] if batch_size >= 2 else query
            tk = self.target_key.to(query.device, query.dtype)
            tk = tk.view(1, -1, attn.heads, head_dim).transpose(1, 2)
            probe_attn = torch.matmul(q_cond * scale, tk.transpose(-2, -1))
            probe_attn = probe_attn.softmax(dim=-1)
            self.store.store_probe(self.layer_name, probe_attn.squeeze(0).detach())
        else:
            hidden_out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )

        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_out = hidden_out.to(query.dtype)
        hidden_out = attn.to_out[0](hidden_out)
        hidden_out = attn.to_out[1](hidden_out)
        return hidden_out


class DualProbeCrossAttnProcessor:
    """
    Computes two probe attentions per forward pass — one for image-contrastive
    keys and one for text-concept keys — storing results in two separate
    AttentionProbeStores. Zero extra UNet calls.
    """

    def __init__(
        self,
        store_image: AttentionProbeStore,
        store_text: AttentionProbeStore,
        layer_name: str,
        target_key_image: torch.Tensor,
        target_key_text: torch.Tensor,
    ):
        self.store_image = store_image
        self.store_text = store_text
        self.layer_name = layer_name
        self.target_key_image = target_key_image
        self.target_key_text = target_key_text

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        enc_states = encoder_hidden_states if is_cross else hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(enc_states)
        value = attn.to_v(enc_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        probe_active = is_cross and (self.store_image.active or self.store_text.active)

        if probe_active:
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = attn_weights.softmax(dim=-1)
            hidden_out = torch.matmul(attn_probs, value)

            q_cond = query[1:2] if batch_size >= 2 else query

            def _probe(target_key, store):
                tk = target_key.to(query.device, query.dtype)
                tk = tk.view(1, -1, attn.heads, head_dim).transpose(1, 2)
                pa = torch.matmul(q_cond * scale, tk.transpose(-2, -1)).softmax(dim=-1)
                store.store_probe(self.layer_name, pa.squeeze(0).detach())

            if self.store_image.active:
                _probe(self.target_key_image, self.store_image)
            if self.store_text.active:
                _probe(self.target_key_text, self.store_text)
        else:
            hidden_out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )

        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_out = hidden_out.to(query.dtype)
        hidden_out = attn.to_out[0](hidden_out)
        hidden_out = attn.to_out[1](hidden_out)
        return hidden_out


def precompute_target_keys(
    unet,
    target_embeds: torch.Tensor,
    target_resolutions: list = [16, 32],
) -> Dict[str, torch.Tensor]:
    """
    Pre-compute K_target = layer.to_k(target_embeds) for each cross-attention
    layer. This is a one-time cost (microseconds).

    Args:
        unet: UNet2DConditionModel
        target_embeds: [1, 77, 768] text encoder output for target concept
        target_resolutions: Only compute for layers at these resolutions

    Returns:
        {layer_name: K_target tensor} dict
    """
    target_keys = {}
    for name, module in unet.named_modules():
        if not name.endswith(".attn2") or not hasattr(module, "to_k"):
            continue
        res = _get_resolution(name)
        if res not in target_resolutions:
            continue
        with torch.no_grad():
            k_target = module.to_k(
                target_embeds.to(module.to_k.weight.device, module.to_k.weight.dtype)
            )
        target_keys[name] = k_target.detach()
    return target_keys


def register_attention_probe(
    unet,
    probe_store: AttentionProbeStore,
    target_keys: Dict[str, torch.Tensor],
    target_resolutions: list = [16, 32],
) -> Dict[str, Any]:
    """
    Replace attn2 processors with ProbeCrossAttnProcessor at target resolutions.

    Returns:
        original_processors dict for restoration via restore_processors()
    """
    original_processors = dict(unet.attn_processors)
    new_processors = {}
    hooked = 0

    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            layer_name = name.replace(".processor", "")
            res = _get_resolution(layer_name)
            if res in target_resolutions and layer_name in target_keys:
                new_processors[name] = ProbeCrossAttnProcessor(
                    store=probe_store,
                    layer_name=layer_name,
                    target_key=target_keys[layer_name],
                )
                hooked += 1
            else:
                new_processors[name] = proc
        else:
            new_processors[name] = proc

    unet.set_attn_processor(new_processors)
    print(f"  [probe] Hooked {hooked} cross-attention layers (res={target_resolutions})")
    return original_processors


def register_dual_attention_probe(
    unet,
    store_image: AttentionProbeStore,
    store_text: AttentionProbeStore,
    target_keys_image: Dict[str, torch.Tensor],
    target_keys_text: Dict[str, torch.Tensor],
    target_resolutions: list = [16, 32],
) -> Dict[str, Any]:
    """
    Register DualProbeCrossAttnProcessor on attn2 layers. Both image and text
    probe maps are collected in one UNet forward pass.

    Returns:
        original_processors dict for restoration
    """
    original_processors = dict(unet.attn_processors)
    new_processors = {}
    hooked = 0

    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            layer_name = name.replace(".processor", "")
            res = _get_resolution(layer_name)
            if (res in target_resolutions
                    and layer_name in target_keys_image
                    and layer_name in target_keys_text):
                new_processors[name] = DualProbeCrossAttnProcessor(
                    store_image=store_image,
                    store_text=store_text,
                    layer_name=layer_name,
                    target_key_image=target_keys_image[layer_name],
                    target_key_text=target_keys_text[layer_name],
                )
                hooked += 1
            else:
                new_processors[name] = proc
        else:
            new_processors[name] = proc

    unet.set_attn_processor(new_processors)
    print(f"  [probe] Hooked {hooked} dual cross-attention layers (res={target_resolutions})")
    return original_processors


def restore_processors(unet, original_processors: Dict[str, Any]):
    """Restore original attention processors."""
    unet.set_attn_processor(original_processors)


def compute_attention_spatial_mask(
    probe_store: AttentionProbeStore,
    token_indices: Optional[List[int]] = None,
    target_resolution: int = 64,
    resolutions_to_use: list = [16, 32],
) -> torch.Tensor:
    """
    Aggregate probe attention maps into a spatial mask.

    Pipeline:
        1. Per layer: avg across heads -> select target tokens -> max across tokens
        2. Group layers by resolution, average within each group
        3. Bilinear upsample all groups to target_resolution (64)
        4. Average across resolution groups
        5. Min-max normalize to [0, 1]

    Args:
        probe_store: AttentionProbeStore with maps from current step
        token_indices: Token positions to use (None = first 19 non-special tokens)
        target_resolution: Output spatial resolution (64 for SD latent space)
        resolutions_to_use: Which resolution groups to aggregate

    Returns:
        Spatial mask of shape [H, W], values in [0, 1]
    """
    maps = probe_store.get_maps()
    if not maps:
        return torch.zeros(target_resolution, target_resolution)

    resolution_groups: Dict[int, list] = {}

    for layer_name, attn_map in maps.items():
        res = _get_resolution(layer_name)
        if res not in resolutions_to_use:
            continue

        spatial = attn_map.shape[1]
        native_res = int(spatial ** 0.5)

        # Average across heads
        avg_map = attn_map.mean(dim=0)  # [spatial, 77]

        # Select target tokens
        if token_indices is not None:
            token_map = avg_map[:, token_indices]
        else:
            token_map = avg_map[:, 1:20]  # first 19 real tokens

        # Max across tokens (union of attended regions)
        spatial_map = token_map.max(dim=-1)[0].view(1, 1, native_res, native_res)

        if res not in resolution_groups:
            resolution_groups[res] = []
        resolution_groups[res].append(spatial_map)

    all_maps = []
    for res, maps_list in resolution_groups.items():
        avg = torch.stack(maps_list).mean(0)
        upsampled = F.interpolate(
            avg.float(),
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=False,
        )
        all_maps.append(upsampled)

    if not all_maps:
        return torch.zeros(target_resolution, target_resolution)

    combined = torch.stack(all_maps).mean(0)

    # Normalize to [0, 1]
    flat = combined.reshape(-1)
    vmin, vmax = flat.min(), flat.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)

    return combined.squeeze(0).squeeze(0)


def find_token_indices(
    prompt: str,
    keywords: List[str],
    tokenizer,
) -> List[int]:
    """
    Find token positions in the tokenized prompt that correspond to keywords.
    Handles BPE multi-subword keywords via subsequence matching.

    Args:
        prompt: The full text prompt
        keywords: List of keywords (e.g. ["weapon", "nude", "naked"])
        tokenizer: CLIPTokenizer

    Returns:
        Sorted list of 0-indexed token positions (BOS at position 0)
    """
    prompt_enc = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    prompt_ids = prompt_enc.input_ids[0].tolist()
    matched_indices = set()

    for keyword in keywords:
        kw_enc = tokenizer(keyword, return_tensors="pt")
        kw_ids = kw_enc.input_ids[0].tolist()
        kw_inner = [
            t for t in kw_ids
            if t != tokenizer.bos_token_id and t != tokenizer.eos_token_id
        ]
        if not kw_inner:
            continue
        for i in range(len(prompt_ids) - len(kw_inner) + 1):
            if prompt_ids[i : i + len(kw_inner)] == kw_inner:
                for j in range(len(kw_inner)):
                    matched_indices.add(i + j)

    return sorted(matched_indices)
