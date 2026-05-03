from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_SEED = 42
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_NUM_IMAGES_PER_PROMPT = 1

MODEL_FAMILIES = ("sd14", "sd3", "flux")

MODEL_SPECS = {
    "sd14": {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "steps": 50,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "scheduler": "DDIM",
        "max_sequence_length": None,
    },
    "sd3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "steps": 28,
        "guidance_scale": 7.0,
        "height": 1024,
        "width": 1024,
        "scheduler": "default-flowmatch",
        "max_sequence_length": 256,
    },
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "steps": 28,
        "guidance_scale": 3.5,
        "height": 512,
        "width": 512,
        "scheduler": "default-flowmatch",
        "max_sequence_length": 512,
    },
}

CONCEPT_SETS = {
    "nudity": ("nudity",),
    "i2p-q16-6": (
        "violence",
        "self-harm",
        "shocking",
        "illegal activity",
        "harassment",
        "hate",
    ),
    "i2p-q16-7": (
        "violence",
        "self-harm",
        "shocking",
        "sexual",
        "illegal activity",
        "harassment",
        "hate",
    ),
    "i2p-paper-11": (
        "hate",
        "harassment",
        "violence",
        "suffering",
        "humiliation",
        "harm",
        "suicide",
        "sexual",
        "nudity",
        "bodily fluids",
        "blood",
    ),
}

PROMPT_COLUMN_CANDIDATES = (
    "prompt",
    "adv_prompt",
    "sensitive prompt",
    "text",
    "caption",
)

SLD_SAFETY_CONCEPT = (
    "hate, harassment, violence, suffering, humiliation, harm, suicide, "
    "sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, "
    "drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
)

SLD_VARIANTS = {
    "SLD-Weak": {
        "sld_guidance_scale": 200,
        "sld_warmup_steps": 15,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "SLD-Medium": {
        "sld_guidance_scale": 1000,
        "sld_warmup_steps": 10,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    },
    "SLD-Strong": {
        "sld_guidance_scale": 2000,
        "sld_warmup_steps": 7,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
    "SLD-Max": {
        "sld_guidance_scale": 5000,
        "sld_warmup_steps": 0,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
}
