"""
AuraFlow / Pony V7 utilities for classifier guidance.

Key differences from PixArt:
  - Velocity prediction (not epsilon)
  - UMT5 text encoder (2048-dim, not T5 4096-dim)
  - LlamaTokenizerFast (not T5Tokenizer)
  - No learned sigma (direct 4ch output, not 8ch)
  - Attention mask baked into embeddings (not passed separately)
  - Timestep normalized to [0, 1] (not integer [0, 1000])
  - FlowMatchEulerDiscreteScheduler (not DDIMScheduler)
"""

import torch


def load_auraflow_components(model_id="purplesmartai/pony-v7-base",
                             device="cuda", dtype=torch.bfloat16):
    """
    Load all AuraFlow / Pony V7 model components.

    Returns:
        dict with keys: vae, transformer, tokenizer, text_encoder, scheduler
    """
    from diffusers import (
        AuraFlowTransformer2DModel,
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
    )
    from transformers import AutoTokenizer, UMT5EncoderModel

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.to(device).eval().requires_grad_(False)

    transformer = AuraFlowTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
    )
    transformer.to(device, dtype=dtype).eval().requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder",
    )
    text_encoder.to(device, dtype=dtype).eval().requires_grad_(False)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler",
    )

    return {
        "vae": vae,
        "transformer": transformer,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "scheduler": scheduler,
    }


def encode_prompt(tokenizer, text_encoder, prompt, device,
                  max_sequence_length=256, dtype=torch.bfloat16):
    """
    Encode text prompt using UMT5 text encoder.

    AuraFlow convention: attention mask is baked into embeddings by zeroing
    out padded positions. The transformer does NOT receive a separate mask.

    Args:
        prompt: str or list[str]

    Returns:
        prompt_embeds: (B, seq_len, 2048) with padded positions zeroed
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )[0].to(dtype=dtype)

    # Bake attention mask into embeddings (AuraFlow convention)
    mask = text_inputs.attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
    prompt_embeds = prompt_embeds * mask.to(prompt_embeds.dtype)

    return prompt_embeds


def auraflow_forward(transformer, latents, timestep, encoder_hidden_states):
    """
    Single forward pass through AuraFlow transformer.

    Unlike PixArt:
      - No learned sigma (output is direct 4ch velocity)
      - No encoder_attention_mask (baked into embeddings)
      - Timestep must be normalized to [0, 1]

    Args:
        transformer: AuraFlowTransformer2DModel
        latents: (B, 4, H, W) noisy latent
        timestep: (B,) tensor of timesteps in [0, 1000] range
        encoder_hidden_states: (B, seq_len, 2048) text embeddings (masked)

    Returns:
        v_pred: (B, 4, H, W) velocity prediction
    """
    # Normalize timestep to [0, 1] as AuraFlow expects
    timestep_norm = timestep.float() / 1000.0
    timestep_norm = timestep_norm.to(device=latents.device, dtype=latents.dtype)

    v_pred = transformer(
        latents,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep_norm,
        return_dict=False,
    )[0]

    return v_pred
