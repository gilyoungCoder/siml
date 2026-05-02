import os 
import argparse
import inspect

import torch

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .encoder_unet_model import EncoderUNetModel, EncoderUNetModelForClassification
from .hierarchical_model import HierarchicalUNetClassifier


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )

def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    in_channels=4,
    out_channels=32,
    hierarchical: bool = True,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 3)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if hierarchical:
        base_args = dict(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=classifier_width,
            out_channels=out_channels,
            num_res_blocks=classifier_depth,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            use_fp16=classifier_use_fp16,
            num_head_channels=64,
            use_scale_shift_norm=classifier_use_scale_shift_norm,
            resblock_updown=classifier_resblock_updown,
            pool=classifier_pool,
        )
        return HierarchicalUNetClassifier(base_args)
    # return EncoderUNetModel(
    return EncoderUNetModelForClassification(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels, #1,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def discriminator_defaults(num_classes=2):
    """
    Defaults for classifier models.
    """
    # discriminator_args = dict(
    #     image_size=8,
    #     classifier_use_fp16=False,
    #     classifier_width=128,
    #     classifier_depth=2,
    #     classifier_attention_resolutions="32,16,8",
    #     classifier_use_scale_shift_norm=True,
    #     classifier_resblock_updown=True,
    #     classifier_pool="attention",
    #     out_channels=1,
    # )
    discriminator_args = dict(
        image_size=128,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="attention",
        out_channels=num_classes,
        in_channels=4,
    )
    return discriminator_args
# 

def create_discriminator(ckpt_path, condition, eval=False, channel=4, num_classes=2, hierarchical=True):
    discriminator_args = discriminator_defaults(num_classes=num_classes)
    discriminator_args["in_channels"] = channel
    # discriminator_args["condition"] = condition
    # For now, the condition for the discriminator is not needed.
    
    discriminator = create_classifier(**discriminator_args, hierarchical=hierarchical)
    return discriminator

def load_discriminator(ckpt_path, condition, eval=False, channel=4, num_classes=2, hierarchical=True):
    discriminator = create_discriminator(ckpt_path, condition, eval, channel, num_classes=num_classes, hierarchical=hierarchical)
    if ckpt_path is not None:
        ckpt_path = os.path.join(os.getcwd(), ckpt_path) if ckpt_path[0] != "/" else ckpt_path 
        discriminator_state = torch.load(ckpt_path, map_location="cpu")
        discriminator.load_state_dict(discriminator_state)
    if eval:
        discriminator.eval()
    return discriminator