import argparse
import logging
import math
import os
import random
import itertools
from pathlib import Path
from typing import Iterable, Optional

import gc
import copy
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

from accelerate import Accelerator

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm


from mmengine.config import Config
from geo_utils.data.missing_person import MissingPersonDataset
from geo_utils.data.missing_person import DummyMissingPersonDataset
from geo_utils.data.new_coco_stuff import NewCOCOStuffDataset
from geo_utils.data.new_new_coco_stuff import NewNewCocoStuffDataset
from geo_utils.data.encoded_coco_stuff import EncodedCocoStuffDataset, EncodedCocoStuffManualDataset
from geo_utils.data.concat_dataset import ConcatDataset, SimpleDataset
from geo_utils.guidance_utils import GuidanceModel
import geo_utils.misc as misc

from geo_models.embed import SplitEmbedding
from geo_models.classifier.classifier import create_classifier, load_discriminator
from geo_models.vaal.vaal_models import VAE, Discriminator

from PIL import Image

import random

logger = get_logger(__name__)

def get_indices_from_dataset(original_data_list, file_name):
    if file_name in original_data_list:
        return original_data_list.index(file_name)
    else:
        return -1


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # model pre-trained name (download from huggingface) or path (local) 
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v1",
        help="Text prompt version. Default to be version3 which is constructed with only camera variables",
    )
    
    parser.add_argument(
        "--num_bucket_per_side",
        type=int,
        default=None,
        nargs="+", 
        help="Location bucket number along each side (i.e., total bucket number = num_bucket_per_side * num_bucket_per_side) ",
    )
    
    parser.add_argument("--bucket_sincos_embed", action="store_true", help="Whether to use 2D sine-cosine embedding for bucket locations")
    
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    
    parser.add_argument(
        "--train_text_encoder_params", 
        type=str, 
        default=["token_embedding", "position", "encoder", "final_layer_norm"], 
        nargs="+", 
        help="token_embedding, position (position_embedding & position_ids), encoder, final_layer_norm, added_embedding (means tuning added tokens while fixing word tokens) "
    )
    
    # foreground loss specifics
    parser.add_argument("--foreground_loss_mode", type=str, default=None, help="None, constant and area")
    
    parser.add_argument("--foreground_loss_weight", type=float, default=1.0, help="Might be utilized differently with respect to loss mode")
    
    parser.add_argument("--foreground_loss_norm", action="store_true", help="Whether to normalize bbox mask")
    
    parser.add_argument("--feat_size", type=int, default=64, help="Feature size of LDMs. Default to be 64 for stable diffusion v1.5")
    
    # unconditional generation specifics
    parser.add_argument("--uncond_prob", type=float, default=0.0, help="Probability to downgrade to unconditional generation")
    
    # data specifics
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    
    # output specifics
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    
    # data augmentation specifics
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    # optimization specifics
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        # default=1e-4,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_text_ratio",
        type=float,
        default=1.0,
        help="Ratio of text encoder LR with respect to UNet LR",
    )
    parser.add_argument(
        "--lr_text_layer_decay",
        type=float,
        default=1.0,
        help="Layer-wise LR decay ratio of text encoder",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--use_ema_text", action="store_true", help="Whether to use EMA model for text encoder.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    # huggingface specifics
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    
    # logging specifics
    # logging_dir = os.path.join(args.output_dir, args.logging_dir)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # BLIP finetune
    parser.add_argument(
        "--no_blip_finetune", 
        action="store_true", 
        help="Whether to finetune with the captions extracted from the BLIP model.")

    # LoRA Finetuning
    parser.add_argument(
        "--lora_mode", action="store_true", help="Whether to use LoRA mode for training.")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA")

    # evaluation specifics
    parser.add_argument("--save_ckpt_freq", type=int, default=10000)

    parser.add_argument("--time_dependent", 
                        action="store_true", help="Whether to use time-dependent discriminator")

    parser.add_argument("--adversarial_scale", type=float, default=1.0, help="Adversarial scale for the discriminator")
    parser.add_argument("--seen_dataset_ratio", type=float, default=0.1, help="Ratio of seen dataset")
    parser.add_argument("--cycle", type=int, default=0, help="The number of cycle")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="The checkpoint to resume training")
    parser.add_argument("--oversample_from_unseen", action="store_true", help="Whether to oversample from unseen dataset")
    parser.add_argument("--for_guidance", action="store_true", help="Whether to use for guidance")
    parser.add_argument("--model_yaml", type=str, default="", help="Model yaml file")
    parser.add_argument("--manual_seen_file_name", type=str, default="", help="Manual seen file name")
    parser.add_argument("--manual_unseen_file_name", type=str, default="", help="Manual unseen file name")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        # torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]
    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()


def check_existence(text, candidates):
    for each in candidates:
        if each in text:
            return True
    return False

# def set_model():
def collate_fn_foreground_loss_mode(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples]).to(memory_format=torch.contiguous_format).long()
    
    bbox_mask = torch.stack([example["bbox_mask"] for example in examples]).unsqueeze(1).float()# [B, 1, H, W]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "bbox_mask": bbox_mask,
    }

def collate_fn_non_foreground_loss_mode(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples]).to(memory_format=torch.contiguous_format).long()
    
    bbox_mask = None # [B, 1, H, W]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "bbox_mask": bbox_mask,
    }

def collate_fn_concatenate(examples):
    collected_img = []
    collected_int = []
    for example in examples:
        for each in example:
            collected_img.append(each[0])
            collected_int.append(each[1])
    collected_img = torch.stack(collected_img)
    collected_int = torch.stack(collected_int)
    return collected_img, collected_int

def main(models=None):
    if os.environ.get("PJRT_DEVICE", None) == "TPU":
        import torch_xla.core.xla_model as xm
        import torch_xla

    args = parse_args()

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # kwargs_handlers=[ddp_kwargs],
    )
    accelerator.print("The device:", accelerator.device)
    accelerator.print("The number of processes:", accelerator.num_processes)
    accelerator.print("State")
    accelerator.print(accelerator.state)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_local_main_process:
        print("{}".format(args).replace(', ', ',\n'))
    

    # If passed along, set the training seed now.
    if args.seed is not None:
        # set_seed(args.seed)
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        print('Set random seed to: {}'.format(seed))

    # Handle the repository creation
    if accelerator.is_local_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    if models is not None:
        text_encoder, vae, unet, tokenizer = models
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
   
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        
    # If used, actual lr = base lr * accum * bs * num_process (base lr corresponds to a single data sample - not the usual 256 based)
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    pipe = None
    if args.model_yaml == "":
        ValueError("Model yaml file is not provided")
    freedom_model_arg_file = args.model_yaml
    # freedom_model_ckpt = "/home/djfelrl11/geodiffusion/work_dirs/missing_person_512_512_240919_disc_seen_unseen_vaal/checkpoint/iter_1380"
    freedom_model_ckpt = None if not args.resume else args.resume_ckpt
    freedom_model = GuidanceModel(pipe, freedom_model_arg_file, freedom_model_ckpt, accelerator.device)
    latent_vae = freedom_model.gradient_model.model["vae"]
    latent_vae_encoder = latent_vae.encoder
    latent_vae_decoder = latent_vae.decoder
    discriminator = freedom_model.gradient_model.model["discriminator"]

    if torch.cuda.is_available() and accelerator.num_processes > 1:
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        latent_vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(latent_vae)


    # params_to_optimize = discriminator.parameters()
    discriminator_parameters = discriminator.parameters()
    # latent_vae_parameters = latent_vae.parameters()
    latent_vae_parameters = itertools.chain(latent_vae_encoder.parameters(), latent_vae_decoder.parameters())
    discriminator_optimizer = optimizer_cls(
        discriminator_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    latent_vae_optimizer = optimizer_cls(
        latent_vae_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer_cls = torch.optim.SGD
    # discriminator_optimizer = optimizer_cls(
    #     discriminator_parameters,
    #     lr=args.learning_rate,
    #     # weight_decay=args.adam_weight_decay,
    # )
    # latent_vae_optimizer = optimizer_cls(
    #     latent_vae_parameters,
    #     lr=args.learning_rate,
    #     # weight_decay=args.adam_weight_decay,
    # )
    
    accelerator.print("Discriminator optimizer:", discriminator_optimizer)
    accelerator.print("Latent VAE optimizer:", latent_vae_optimizer)
    # if accelerator.is_local_main_process:
        # print(optimizer)
        

    # MODIFY: to be consistent with the pre-trained model
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column


    # Before loading datasets, check the additional seen dataset,
    # If there are jpg files in the directory, convert them into latent code as pt file with flip
    convert_jpt_to_npy_transform_flip = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.RandomHorizontalFlip(1),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    convert_jpt_to_npy_transform_no_flip = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def convert_jpg_into_npy(current_dir):
        child_files = os.listdir(current_dir)
        child_files = tqdm(child_files)
        for each in child_files:
            if os.path.isdir(os.path.join(current_dir, each)):
                convert_jpg_into_npy(os.path.join(current_dir, each))
            elif os.path.isfile(os.path.join(current_dir, each)) and each.endswith(".jpg"):
                img = Image.open(os.path.join(current_dir, each)).convert('RGB')
                flip_img = convert_jpt_to_npy_transform_flip(img)
                flip_img = flip_img.unsqueeze(0)
                flip_img = flip_img.to(accelerator.device)
                with torch.no_grad():
                    flip_z = vae.encode(flip_img)
                    flip_z = flip_z.latent_dist.sample()
                    flip_z = flip_z.cpu()
                    flip_z = flip_z.numpy()
                
                no_flip_img = convert_jpt_to_npy_transform_no_flip(img)
                no_flip_img = no_flip_img.unsqueeze(0)
                no_flip_img = no_flip_img.to(accelerator.device)
                with torch.no_grad():
                    no_flip_z = vae.encode(no_flip_img)
                    no_flip_z = no_flip_z.latent_dist.sample()
                    no_flip_z = no_flip_z.cpu()
                    no_flip_z = no_flip_z.numpy()
                
                if os.environ.get("PJRT_DEVICE", None) == "TPU":
                    xm.mark_step()
                # torch.save(no_flip_z, os.path.join(current_dir, each.replace(".jpg", ".pt")))
                # torch.save(flip_z, os.path.join(current_dir, each.replace(".jpg", "_flip.pt")))
                np.save(os.path.join(current_dir, each.replace(".jpg", ".npy")), no_flip_z)
                np.save(os.path.join(current_dir, each.replace(".jpg", "_flip.npy")), flip_z)
                os.remove(os.path.join(current_dir, each))

    # if accelerator.is_local_main_process:
    #     print("convert jpg to npy in additional seen dataset")
    #     convert_jpg_into_npy("/home/djfelrl11/additional_seen_dataset")
    #     print("done")
    # accelerator.wait_for_everyone()

    num_workers = 4 * accelerator.num_processes
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    random.seed(0)
    dataset_cfg = Config.fromfile(args.dataset_config_name)
    data_path = os.path.join(dataset_cfg.data_root, dataset_cfg.data.train.img_prefix)
    total_train_dataset_len = len(os.listdir(data_path))

    print(args.seen_dataset_ratio)

    initial_seen_indice = random.sample(range(total_train_dataset_len), int(total_train_dataset_len * args.seen_dataset_ratio))

    if accelerator.is_local_main_process:
        initial_seen_log = []
        dataset_index_to_file = sorted(os.listdir(data_path))
        for each in initial_seen_indice:
            initial_seen_log.append(dataset_index_to_file[each])
        with open("initial_seen_file.txt", "w") as f:
            for each in initial_seen_log:
                f.write(each + "\n")

    real_additional_dataset_paths = []
    real_additional_seen_dataset_yaml_dir = "/home/djfelrl11/additional_seen_dataset_yaml"
    for each in os.listdir(real_additional_seen_dataset_yaml_dir):
        with open(os.path.join(real_additional_seen_dataset_yaml_dir, each), "r") as f:
            if ".yaml" in each:
                additional_dataset_dict = yaml.load(f, Loader=yaml.FullLoader)
                real_additional_dataset_paths += [os.path.join(data_path, filename) for filename in additional_dataset_dict.keys()]
            elif ".txt" in each:
                additional_dataset_list = f.readlines()
                additional_dataset_list = [ additional_dataset.strip() for additional_dataset in additional_dataset_list]
                real_additional_dataset_paths +=[os.path.join(data_path, filename) for filename in additional_dataset_list]
    
    generated_additional_dataset_dir = "/home/djfelrl11/additional_seen_dataset"
    generated_additional_dataset_paths = []
    # for each in os.listdir(generated_additional_dataset_dir):
    #     if os.path.isdir(os.path.join(generated_additional_dataset_dir, each)):
    #         for each_file in os.listdir(os.path.join(generated_additional_dataset_dir, each)):
    #             generated_additional_dataset_paths.append(os.path.join(generated_additional_dataset_dir, each, each_file))
    #     elif os.path.isfile(os.path.join(generated_additional_dataset_dir, each)) and each.endswith(".npy"):
    #         generated_additional_dataset_paths.append(os.path.join(generated_additional_dataset_dir, each))
    #     else:
    #         ValueError("Invalid file")
    # Get all npy file in the directory recursively
    for root, dirs, files in os.walk(generated_additional_dataset_dir):
        for each in files:
            if each.endswith(".npy"):
                generated_additional_dataset_paths.append(os.path.join(root, each))
    
    if args.oversample_from_unseen:
        additional_dataset_paths = real_additional_dataset_paths + generated_additional_dataset_paths
        total_seen_dataset_len = len(initial_seen_indice) + len(additional_dataset_paths)

        # load entropy of unseen dataset 
        with open("/home/djfelrl11/geodiffusion/coco_2017_cycle_0_entropy.yaml", "r") as f:
            unseen_entropy = yaml.load(f, Loader=yaml.FullLoader)
            unseen_entropy_sorted = dict(sorted(unseen_entropy.items(), key=lambda x: x[1]))
        
        oversample_dataset_len = (len(unseen_entropy_sorted) + len(generated_additional_dataset_paths)) // 2 - total_seen_dataset_len
        data_path_list = sorted(os.listdir(data_path))
        initial_seen_path = [data_path_list[each] for each in initial_seen_indice]
        count = 0
        for key, value in unseen_entropy_sorted.items():
            if count == oversample_dataset_len:
                break
            key_full_path = os.path.join(data_path, key)
            if key_full_path not in real_additional_dataset_paths and key not in initial_seen_path:
                real_additional_dataset_paths.append(key_full_path)
                count += 1
            

    additional_dataset_paths = real_additional_dataset_paths + generated_additional_dataset_paths

    # original_seen_dataset_len = len(initial_seen_indice) + len(additional_dataset_paths)
    original_seen_dataset_len = len(initial_seen_indice) + len(real_additional_dataset_paths) + \
        len(generated_additional_dataset_paths) // 2

    dataset_list = [os.path.join(data_path, each) for each in sorted(os.listdir(data_path))]
    unseen_exclude_file_name = real_additional_dataset_paths
    unseen_exclude_indices = []
    for each in unseen_exclude_file_name:
        index = get_indices_from_dataset(dataset_list, each)
        if index != -1:
            unseen_exclude_indices.append(index)
    print("len of unseen_exclude_indices", len(unseen_exclude_indices))


    print("initial seen indices len", len(initial_seen_indice))
    print("after initial seen ", len(set(initial_seen_indice) - set(unseen_exclude_indices)))
    assert set(initial_seen_indice) - set(unseen_exclude_indices) == set(initial_seen_indice), "seen and unseen overlap"

    total_unseen_indices = list(set(range(total_train_dataset_len)) - set(initial_seen_indice))
    unseen_indices = random.sample(total_unseen_indices, len(total_unseen_indices) // 2)

    if args.for_guidance:
        # Mutual exclusive unseen_indices
        unseen_indices = list(set(range(total_train_dataset_len)) - set(initial_seen_indice) - set(unseen_indices))

    
    unseen_indices = list(set(unseen_indices) - set(unseen_exclude_indices))
    unseen_indices = random.sample(unseen_indices, original_seen_dataset_len) if not args.oversample_from_unseen else unseen_indices

    if accelerator.is_local_main_process:
        initial_seen_log = {}
        dataset_index_to_file = sorted(os.listdir(data_path))
        for each in unseen_indices:
            # initial_seen_log.append(dataset_index_to_file[each])
            initial_seen_log[dataset_index_to_file[each]] = each
        file_name_suffix = "guidance" if args.for_guidance else "vaal"
        file_name = f"initial_unseen_file_{file_name_suffix}.yaml"
        # with open("initial_unseen_file.yaml", "w") as f:
        #     yaml.dump(initial_seen_log, f)
        with open(file_name, "w") as f:
            yaml.dump(initial_seen_log, f)

    # Balance as 50 50
    seen_indices = initial_seen_indice

    # seen_dataset = NewNewCocoStuffDataset(data_path, indices=seen_indices, 
    #                                         additional_dataset_paths=additional_dataset_paths, transform=transform)
    # unseen_dataset = NewNewCocoStuffDataset(data_path, indices=unseen_indices, transform=transform)

    # seen_dataset = EncodedCocoStuffDataset(data_path, indices=seen_indices,
    #                                         additional_dataset_paths=additional_dataset_paths, transform=transform, epoch=args.num_train_epochs)
    # unseen_dataset = EncodedCocoStuffDataset(data_path, indices=unseen_indices, transform=transform, epoch=args.num_train_epochs)

    if args.manual_seen_file_name != "":
        seen_dataset = EncodedCocoStuffManualDataset(
            data_path, manual_data_file=args.manual_seen_file_name, transform=transform, epoch=1)
    else:
        seen_dataset = EncodedCocoStuffDataset(data_path, indices=seen_indices,
                                            additional_dataset_paths=additional_dataset_paths, transform=transform, epoch=1)
    if args.manual_unseen_file_name != "":
        unseen_dataset = EncodedCocoStuffManualDataset(
            data_path, manual_data_file=args.manual_unseen_file_name, transform=transform, epoch=1)
    else:
        unseen_dataset = EncodedCocoStuffDataset(data_path, indices=unseen_indices, transform=transform, epoch=1)

    
    seen_dataloader = torch.utils.data.DataLoader(
        seen_dataset, batch_size=args.train_batch_size,
        drop_last=True, num_workers=num_workers
    )

    unseen_dataloader = torch.utils.data.DataLoader(
        unseen_dataset, batch_size=args.train_batch_size,
        drop_last=True, num_workers=num_workers
    )
    

    val_seen_idx = random.sample(range(len(seen_dataset)), int(len(seen_dataset) // args.num_train_epochs * 0.1))
    val_seen_dataset = torch.utils.data.Subset(seen_dataset, val_seen_idx)
    val_unseen_idx = random.sample(range(len(unseen_dataset)), int(len(unseen_dataset) // args.num_train_epochs * 0.1))
    val_unseen_dataset = torch.utils.data.Subset(unseen_dataset, val_unseen_idx)

    val_seen_dataloader = torch.utils.data.DataLoader(
        val_seen_dataset, shuffle=False, batch_size=args.train_batch_size,
        drop_last=False, num_workers=num_workers
    )
    val_unseen_dataloader = torch.utils.data.DataLoader(
        val_unseen_dataset, shuffle=False, batch_size=args.train_batch_size,
        drop_last=False, num_workers=num_workers
    )
    accelerator.print("len of initial Seen dataset: ", len(initial_seen_indice))
    accelerator.print("len of total Seen dataset: ", len(seen_dataset))
    accelerator.print("len of Unseen dataset: ", len(unseen_dataset))
    
    ##################################
    # Prepare dataloader
    ##################################
    
    if args.foreground_loss_mode:
        collate_fn = collate_fn_foreground_loss_mode
    else:
        collate_fn = collate_fn_non_foreground_loss_mode

    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(min(len(seen_dataloader), len(unseen_dataloader)) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        # args.max_train_steps = (args.num_train_epochs * num_update_steps_per_epoch) // accelerator.num_processes
        args.max_train_steps = num_update_steps_per_epoch // accelerator.num_processes # The epoch is included in the dataset, so no need to count
    
    
    # discriminator, latent_vae, discriminator_optimizer, latent_vae_optimizer, seen_dataloader, unseen_dataloader = accelerator.prepare(
    #     discriminator, latent_vae, discriminator_optimizer, latent_vae_optimizer, seen_dataloader, unseen_dataloader
    # )

    discriminator, latent_vae_encoder, latent_vae_decoder, discriminator_optimizer, latent_vae_optimizer, seen_dataloader, unseen_dataloader = accelerator.prepare(
        discriminator, latent_vae_encoder, latent_vae_decoder, discriminator_optimizer, latent_vae_optimizer, seen_dataloader, unseen_dataloader
    )


    dataloader_len = max(len(seen_dataloader), len(unseen_dataloader))
    val_seen_dataloader, val_unseen_dataloader = accelerator.prepare(val_seen_dataloader, val_unseen_dataloader)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print("weight_dtype", weight_dtype)
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    vae = vae.to(device=accelerator.device)
    vae = vae.to(dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_local_main_process:
        saved_args = copy.copy(args)
        saved_args.num_bucket_per_side = ' '.join([str(each) for each in saved_args.num_bucket_per_side])
        saved_args.train_text_encoder_params = ' '.join(saved_args.train_text_encoder_params)
        init_kwargs = {"wandb":{"name":args.output_dir}}
        accelerator.init_trackers("text2image-fine-tune", config=vars(saved_args), init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num examples = {len(unseen_dataset) + len(seen_dataset)}")
    logger.info(f"  Num examples = {dataloader_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    if args.resume:
        global_step = int(args.resume_ckpt.split("_")[-1])
        # progress_bar.update(global_step)
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, initial=global_step)
        accelerator.print("Resuming from step", global_step)
    else:
        global_step = 0
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        accelerator.print("Starting from step", global_step)
    progress_bar.set_description("Steps")


    # Check the accelerator environment and set the autocast device
    autocast_device = "cpu"
    if os.environ.get("PJRT_DEVICE", None) == "TPU":
        autocast_device = "xla"
    elif torch.cuda.is_available():
        autocast_device = "cuda"

    # loss = nn.BCEWithLogitsLoss()

    # loss_function = nn.BCELoss()
    mse_loss = nn.MSELoss()
    # bce_loss = nn.BCELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    def vae_loss_fn(x, recon, mu, logvar, beta):
        MSE = torch.sum((recon - x) ** 2, dtype=recon.dtype)
        KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar), dtype=logvar.dtype)
        KLD = KLD * beta

        loss = MSE + KLD
        loss = loss / torch.numel(x)
        return loss
    
    def disc_loss_fn(real_logit, fake_logit, real_label, fake_label):
        return bce_loss(fake_logit, fake_label) + bce_loss(real_logit, real_label)

    def set_require_grad(model, flag):
        for param in model.parameters():
            param.requires_grad = flag

    def encoder_output(latent_vae_encoder, latents, timesteps_model_input=None):
        z = latent_vae_encoder(latents, timesteps_model_input)
        if isinstance(z, tuple):
            mu, logvar = z
        else:
            mu, logvar = torch.chunk(z, 2, dim=1)
        return mu, logvar

    def reparameterize(mu, logvar):
        stds = torch.exp(0.5 * logvar)
        epsilon = torch.randn(*mu.size(), dtype=mu.dtype, device=mu.device)
        latents = epsilon * stds + mu
        return latents

    def decoder_output(latent_vae_decoder, z, timesteps_model_input=None):
        x_recon = latent_vae_decoder(z, timesteps_model_input)
        return x_recon
    
    def vae_output(latent_vae_encoder, latent_vae_decoder, latents, timesteps_model_input=None):
        mu, logvar = encoder_output(latent_vae_encoder, latents, timesteps_model_input)
        z = reparameterize(mu, logvar)
        x_recon = decoder_output(latent_vae_decoder, z, timesteps_model_input)
        return x_recon, z, mu, logvar

    def model_output(seen_batch, unseen_batch, is_noisy=False, timesteps=None):
        seen_pixel_values = seen_batch[0].to(weight_dtype)
        # seen_latents = vae.encode(seen_pixel_values).latent_dist.sample()
        seen_latents = seen_pixel_values * 0.18215

        bsz = seen_pixel_values.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=seen_latents.device)
            timesteps = timesteps.long()
        else:
            timesteps = torch.full((bsz,), timesteps)
            timesteps = timesteps.to(seen_latents.device)
            timesteps = timesteps.long()
        timesteps_model_input = timesteps / noise_scheduler.num_train_timesteps
        timesteps_model_input = timesteps_model_input.float()

        if is_noisy:
            seen_noise = torch.randn_like(seen_latents)
            # Sample a random timestep for each image
            seen_noisy_latents = noise_scheduler.add_noise(seen_latents, seen_noise, timesteps)
            x_recon_seen, z_seen, mu_seen, logvar_seen = vae_output(latent_vae_encoder, latent_vae_decoder, seen_noisy_latents, timesteps_model_input)
            disc_pred_seen = discriminator(mu_seen, timesteps_model_input)
        else:
            x_recon_seen, z_seen, mu_seen, logvar_seen = vae_output(latent_vae_encoder, latent_vae_decoder, seen_latents)
            mu_seen_clone = mu_seen.clone()
            disc_pred_seen = discriminator(mu_seen_clone)

        unseen_pixel_values = unseen_batch[0].to(weight_dtype)
        # unseen_latents = vae.encode(unseen_pixel_values).latent_dist.sample().to(weight_dtype)
        unseen_latents = unseen_pixel_values * 0.18215
        if is_noisy:
            unseen_noisy_latents = noise_scheduler.add_noise(unseen_latents, seen_noise, timesteps)
            # x_recon_unseen, z_unseen, mu_unseen, logvar_unseen = latent_vae(unseen_noisy_latents, timesteps)
            x_recon_unseen, z_unseen, mu_unseen, logvar_unseen = vae_output(latent_vae_encoder, latent_vae_decoder, unseen_noisy_latents, timesteps_model_input)
            disc_pred_unseen = discriminator(mu_unseen, timesteps_model_input)
        else:
            # x_recon_unseen, z_unseen, mu_unseen, logvar_unseen = latent_vae(unseen_latents)
            x_recon_unseen, z_unseen, mu_unseen, logvar_unseen = vae_output(latent_vae_encoder, latent_vae_decoder, unseen_latents)
            mu_unseen_clone = mu_unseen.clone()
            disc_pred_unseen = discriminator(mu_unseen_clone)

        return seen_latents, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
            unseen_latents, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen

    
    # def validatation(val_seen_dataloader, val_unseen_dataloader, latent_vae, discriminator, weight_dtype):
    def validation(val_seen_dataloader, val_unseen_dataloader, weight_dtype):
        discriminator.eval()
        latent_vae.eval()
        set_require_grad(discriminator, False)
        set_require_grad(latent_vae, False)
        val_seen_dataloader_enumerator = tqdm(enumerate(val_seen_dataloader), disable=not accelerator.is_local_main_process, total=len(val_seen_dataloader))
        val_unseen_dataloader_enumerator = tqdm(enumerate(val_unseen_dataloader), disable=not accelerator.is_local_main_process, total=len(val_unseen_dataloader))
        count = 0 
        if args.time_dependent:
            timestep_interval = 200 # the number that can divide 1000
            val_loss = 0.0
            # val_correct = torch.zeros(noise_scheduler.num_train_timesteps // 10 + 1, )
            val_loss = 0.0
            # val_real_correct = 0
            # val_fake_correct = 0
            val_real_correct = torch.zeros(noise_scheduler.num_train_timesteps // timestep_interval, )
            val_fake_correct = torch.zeros(noise_scheduler.num_train_timesteps // timestep_interval, )
            seen_count = torch.zeros(noise_scheduler.num_train_timesteps // timestep_interval, )
            unseen_count = torch.zeros(noise_scheduler.num_train_timesteps // timestep_interval, )
            seen_loss = torch.zeros(noise_scheduler.num_train_timesteps // timestep_interval, )
            unseen_loss = torch.zeros(noise_scheduler.num_train_timesteps // timestep_interval, )
            # for val_step, (val_seen_batch, val_unseen_batch) in enumerator:
            for val_step, val_seen_batch in val_seen_dataloader_enumerator:
                for timestep_idx in range(0, noise_scheduler.num_train_timesteps, timestep_interval):
                    latent_seen, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
                    latent_unseen, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen = model_output(
                        val_seen_batch, val_seen_batch, args.time_dependent, timesteps=timestep_idx)
                    
                    disc_pred_seen = disc_pred_seen.float()
                    disc_pred_unseen = disc_pred_unseen.float()
                    

                    real_label = torch.ones_like(disc_pred_seen).to(weight_dtype)
                    fake_label = torch.zeros_like(disc_pred_seen).to(weight_dtype)

                    # disc_loss = disc_loss_fn(disc_pred_seen, disc_pred_unseen, real_label, fake_label)
                    # loss = disc_loss
                    # # loss = loss.mean()

                    if os.environ.get("PJRT_DEVICE", None) == "TPU":
                        xm.mark_step()
                    loss = disc_loss_fn(disc_pred_seen, fake_label, real_label, fake_label)
                    loss = loss.sum()
                    seen_loss[timestep_idx // timestep_interval] += loss.detach().item()

                    val_real_correct[timestep_idx // timestep_interval] += ((F.sigmoid(disc_pred_seen) > 0.5).float() == real_label).sum().item()
                    seen_count[timestep_idx // timestep_interval] += disc_pred_seen.shape[0]
            for val_step, val_unseen_batch in val_unseen_dataloader_enumerator:
                for timestep_idx in range(0, noise_scheduler.num_train_timesteps, timestep_interval):
                    latent_seen, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
                    latent_unseen, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen = model_output(
                        val_unseen_batch, val_unseen_batch, args.time_dependent, timesteps=timestep_idx)
                    
                    disc_pred_seen = disc_pred_seen.float()
                    disc_pred_unseen = disc_pred_unseen.float()

                    real_label = torch.ones_like(disc_pred_seen).to(weight_dtype)
                    fake_label = torch.zeros_like(disc_pred_seen).to(weight_dtype)

                    # disc_loss = disc_loss_fn(disc_pred_seen, disc_pred_unseen, real_label, fake_label)
                    # loss = disc_loss
                    # # loss = loss.mean()

                    if os.environ.get("PJRT_DEVICE", None) == "TPU":
                        xm.mark_step()

                    loss = disc_loss_fn(real_label, disc_pred_unseen, real_label, fake_label)
                    loss = loss.sum()
                    unseen_loss[timestep_idx // timestep_interval] += loss.detach().item()

                    val_fake_correct[timestep_idx // timestep_interval] += ((F.sigmoid(disc_pred_unseen) > 0.5).float() == fake_label).sum().item()
                    unseen_count[timestep_idx // timestep_interval] += disc_pred_seen.shape[0]
            
        else:
            val_loss = 0.0
            # val_correct = 0
            val_real_correct = 0
            val_fake_correct = 0
            seen_count = 0
            unseen_count = 0
            seen_loss = 0
            unseen_loss = 0
            for val_step, val_seen_batch in val_seen_dataloader_enumerator:
                latent_seen, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
                latent_unseen, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen = model_output(
                    val_seen_batch, val_seen_batch, args.time_dependent)

                real_label = torch.ones_like(disc_pred_seen).to(weight_dtype)
                fake_label = torch.zeros_like(disc_pred_seen).to(weight_dtype)

                disc_pred_seen = disc_pred_seen.float()
                disc_pred_unseen = disc_pred_unseen.float()

                if os.environ.get("PJRT_DEVICE", None) == "TPU":
                    xm.mark_step()
                loss = disc_loss_fn(real_label, disc_pred_unseen, real_label, fake_label)
                # val_loss += loss.detach().item()
                seen_loss += loss.detach().item()
                val_real_correct += ((F.sigmoid(disc_pred_seen) > 0.5).float() == real_label).sum().item()
                seen_count += disc_pred_seen.shape[0]

            for val_step, val_unseen_batch in val_unseen_dataloader_enumerator:
                latent_seen, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
                latent_unseen, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen = model_output(
                    val_unseen_batch, val_unseen_batch, args.time_dependent)

                disc_pred_seen = disc_pred_seen.float()
                disc_pred_unseen = disc_pred_unseen.float()

                real_label = torch.ones_like(disc_pred_seen).to(weight_dtype)
                fake_label = torch.zeros_like(disc_pred_seen).to(weight_dtype)
                loss = disc_loss_fn(real_label, disc_pred_unseen, real_label, fake_label)

                if os.environ.get("PJRT_DEVICE", None) == "TPU":
                    xm.mark_step()
                # val_loss += loss.detach().item()
                unseen_loss += loss.detach().item()
                val_fake_correct += ((F.sigmoid(disc_pred_unseen) > 0.5).float() == fake_label).sum().item()
                unseen_count += disc_pred_unseen.shape[0]
        
        # accelerator.print("Validation overall batch size: ", count)
        accelerator.print("Validation Seen dataset size: ", seen_count)
        accelerator.print("Validation Unseen dataset size: ", unseen_count)
        # accelerator.print("Validation loss: ", val_loss)
        accelerator.print("Validation real accuracy: ", val_real_correct / seen_count)
        accelerator.print("Validation fake accuracy: ", val_fake_correct / unseen_count)
        accelerator.print("Validation accuracy: ", (val_real_correct + val_fake_correct) / (seen_count + unseen_count))
        if args.time_dependent:
            accelerator.print("Validation real overall accuracy: ", torch.mean(val_real_correct / count))
            accelerator.print("Validation fake overall accuracy: ", torch.mean(val_fake_correct / count))
        accelerator.print("Validation seen loss: ", seen_loss / seen_count)
        accelerator.print("Validation unseen loss: ", unseen_loss / unseen_count)
        discriminator.train()
        latent_vae.train()
        set_require_grad(discriminator, True)
        set_require_grad(latent_vae, True)

    # for epoch in range(args.num_train_epochs):
    # epoch = 0
    # while True:
    for epoch in range(args.num_train_epochs):
    # for seen_batch, unseen_batch in zip(seen_dataloader, unseen_dataloader):
        progress_bar = tqdm(range(dataloader_len), disable=not accelerator.is_local_main_process)
        for seen_batch, unseen_batch in zip(seen_dataloader, unseen_dataloader):
            # First, vae training
            latent_vae_optimizer.zero_grad()
            latent_vae.train()
            discriminator.eval()

            latent_seen, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
            latent_unseen, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen \
                = model_output(seen_batch, unseen_batch, args.time_dependent)

            real_label = torch.ones_like(disc_pred_seen).to(disc_pred_seen.device)
            fake_label = torch.ones_like(disc_pred_seen).to(disc_pred_seen.device)

            unsupervised_vae_loss = vae_loss_fn(latent_seen, x_recon_seen, mu_seen, logvar_seen, 1.0)
            transductive_vae_loss = vae_loss_fn(latent_unseen, x_recon_unseen,  mu_unseen, logvar_unseen, 1.0)

            disc_loss = disc_loss_fn(disc_pred_seen, disc_pred_unseen, real_label, fake_label)

            loss = unsupervised_vae_loss + transductive_vae_loss

            loss = loss.mean()
            accelerator.backward(loss)
            latent_vae_optimizer.step()
            vae_step_loss = loss.detach().item()

            if os.environ.get("PJRT_DEVICE", None) == "TPU":
                xm.mark_step()

            # Second, discriminator training
            discriminator.zero_grad()
            latent_vae.eval()
            discriminator.train()

            seen_batch = copy.deepcopy(seen_batch)
            latent_seen, x_recon_seen, z_seen, mu_seen, logvar_seen, disc_pred_seen, \
            latent_unseen, x_recon_unseen, z_unseen, mu_unseen, logvar_unseen, disc_pred_unseen \
                = model_output(seen_batch, unseen_batch, args.time_dependent)
            
            disc_pred_seen = disc_pred_seen.float()
            disc_pred_unseen = disc_pred_unseen.float()

            real_label = torch.ones_like(disc_pred_seen).to(disc_pred_seen.device).to(disc_pred_seen.dtype)
            fake_label = torch.zeros_like(disc_pred_seen).to(disc_pred_seen.device).to(disc_pred_seen.dtype)

            disc_loss = disc_loss_fn(disc_pred_seen, disc_pred_unseen, real_label, fake_label)
            loss = disc_loss
            loss = loss.mean()

            # Backpropagate
            discriminator_optimizer.zero_grad()
            accelerator.backward(loss)
            discriminator_optimizer.step()
            disc_step_loss = loss.detach().item()

            if os.environ.get("PJRT_DEVICE", None) == "TPU":
                xm.mark_step()
            # Gather the losses across all processes for logging (if we use distributed training).

            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            global_step += 1

            logs = {
                "vae_step_loss": vae_step_loss,
                "disc_step_loss": disc_step_loss,
                "epoch": global_step // (dataloader_len // args.num_train_epochs),
                "step": global_step,
            }

            if autocast_device == "xla":
                tpu_mem_info = xm.get_memory_info(xm.xla_device())
                tpu_mem_used_mb = tpu_mem_info["bytes_used"] // 1024 // 1024
                logs.update({"mem_mb": tpu_mem_used_mb})

            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

    # if global_step >= args.max_train_steps:
    #     break

    # save current checkpoint every 500 iterations
            if global_step % args.save_ckpt_freq == 1:
                accelerator.wait_for_everyone()
                accelerator.print("Save start")
                # Save peft
                if accelerator.is_local_main_process:
                    discriminator_unwrap = accelerator.unwrap_model(discriminator)
                    discriminator_cpu = copy.deepcopy(discriminator_unwrap).to("cpu")
                    discriminator_state_dict = discriminator_cpu.state_dict()

                    latent_vae_unwrap = accelerator.unwrap_model(latent_vae)
                    vae_cpu = copy.deepcopy(latent_vae_unwrap).to("cpu")
                    vae_state_dict = vae_cpu.state_dict()

                    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint', 'iter_' + str(global_step))
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(discriminator_state_dict, os.path.join(checkpoint_dir, "discriminator.pth"))
                    torch.save(vae_state_dict, os.path.join(checkpoint_dir, "vae.pth"))

                accelerator.print("save done")
                accelerator.wait_for_everyone()

                # Validation
                # validation(val_seen_dataloader, val_unseen_dataloader, weight_dtype)


    # Create the pipeline using the trained modules and save it.
    # EMA model: not used during training, just record it and save it finally
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        discriminator_unwrap = accelerator.unwrap_model(discriminator)
        discriminator_cpu = copy.deepcopy(discriminator_unwrap).to("cpu")
        discriminator_state_dict = discriminator_cpu.state_dict()

        # latent_vae_unwrap = accelerator.unwrap_model(latent_vae)
        # vae_cpu = copy.deepcopy(latent_vae_unwrap).to("cpu")
        latent_vae_encoder_unwrap = accelerator.unwrap_model(latent_vae_encoder)
        latent_vae_decoder_unwrap = accelerator.unwrap_model(latent_vae_decoder)
        latent_vae.encoder = latent_vae_encoder_unwrap
        latent_vae.decoder = latent_vae_decoder_unwrap
        vae_cpu = copy.deepcopy(latent_vae).to("cpu")
        vae_state_dict = vae_cpu.state_dict()

        checkpoint_dir = os.path.join(args.output_dir, 'checkpoint', 'iter_' + str(global_step))
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(discriminator_state_dict, os.path.join(checkpoint_dir, "discriminator.pth"))
        torch.save(vae_state_dict, os.path.join(checkpoint_dir, "vae.pth"))
    
    # validation
    # validation(val_seen_dataloader, val_unseen_dataloader, weight_dtype)
    validation(seen_dataloader, unseen_dataloader, weight_dtype)

    if args.push_to_hub:
        repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()

