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
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import convert_state_dict_to_diffusers

from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict


from mmengine.config import Config
from geo_utils.data.missing_person import MissingPersonDataset
from geo_utils.data.missing_person import DummyMissingPersonDataset
from geo_utils.data.new_coco_stuff import NewCOCOStuffDataset
from geo_utils.data.new_new_coco_stuff import NewNewCocoStuffDataset
from geo_utils.data.concat_dataset import ConcatDataset, SimpleDataset
import geo_utils.misc as misc

from geo_models.embed import SplitEmbedding
from geo_models.classifier.classifier import create_classifier, load_discriminator
# if os.environ.get("PJRT_DEVICE", None)=="TPU":
#     import torch_xla.core.xla_model as xm

# import torch_xla.debug.metrics as met

import random
import yaml

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
        default=1e-4,
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

    parser.add_argument("--seen_dataset_ratio", type=float, default=0.1, help="Ratio of seen dataset")
    parser.add_argument("--cycle", type=int, default=0, help="The number of cycle")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="The checkpoint to resume training")
    parser.add_argument("--oversample_from_unseen", action="store_true", help="Whether to oversample from unseen dataset")
    parser.add_argument("--for_guidance", action="store_true", help="Whether to use for guidance")
    
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
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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
    
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if "token_embedding" in args.train_text_encoder_params and "added_embedding" in args.train_text_encoder_params:
        raise ValueError(
            "Added_embedding suggests only tuning added tokens while fixing existing word tokens, which contradicts against token_embedding. "
        )

    if not args.train_text_encoder and args.use_ema_text:
        raise ValueError(
            "Use EMA model for text encoder only when we train text encoder. "
        )

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
    
    discriminator = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4)

    params_to_optimize = discriminator.parameters()
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if accelerator.is_local_main_process:
        print(optimizer)



    # MODIFY: to be consistent with the pre-trained model
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

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
            additional_dataset_dict = yaml.load(f, Loader=yaml.FullLoader)
            real_additional_dataset_paths += [os.path.join(data_path, filename) for filename in additional_dataset_dict.keys()]
    generated_additional_dataset_dir = "/home/djfelrl11/additional_seen_dataset"
    generated_additional_dataset_paths = []
    for each in os.listdir(generated_additional_dataset_dir):
        if os.path.isdir(os.path.join(generated_additional_dataset_dir, each)):
            for each_file in os.listdir(os.path.join(generated_additional_dataset_dir, each)):
                generated_additional_dataset_paths.append(os.path.join(generated_additional_dataset_dir, each, each_file))
        elif os.path.isfile(os.path.join(generated_additional_dataset_dir, each)) and each.endswith(".jpg"):
            generated_additional_dataset_paths.append(os.path.join(generated_additional_dataset_dir, each))
        else:
            ValueError("Invalid file")
    
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

    original_seen_dataset_len = len(initial_seen_indice) + len(additional_dataset_paths)
    # seen_dataset = NewNewCocoStuffDataset(data_path, indices=initial_seen_indice, 
    #                                         additional_dataset_paths=additional_dataset_paths, transform=transform)

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
    
    unseen_indices = list(set(range(total_train_dataset_len)) - set(initial_seen_indice) - set(unseen_exclude_indices))
    unseen_indices = random.sample(unseen_indices, original_seen_dataset_len) if not args.oversample_from_unseen else unseen_indices

    if args.for_guidance:
        # Mutual exclusive unseen_indices
        unseen_indices = list(set(range(total_train_dataset_len)) - set(initial_seen_indice) - set(unseen_exclude_indices) - set(unseen_indices))
        unseen_indices = random.sample(unseen_indices, original_seen_dataset_len) if not args.oversample_from_unseen else unseen_indices

    if accelerator.is_local_main_process:
        initial_seen_log = {}
        dataset_index_to_file = sorted(os.listdir(data_path))
        for each in unseen_indices:
            # initial_seen_log.append(dataset_index_to_file[each])
            initial_seen_log[dataset_index_to_file[each]] = each
        with open("initial_unseen_file.yaml", "w") as f:
            yaml.dump(initial_seen_log, f)
    # unseen_indices = random.sample(unseen_indices, len(seen_dataset)) if not args.oversample_from_unseen else unseen_indices

    # Balance as 50 50
    seen_indices = initial_seen_indice
    # remained_real_inices = list(set(range(total_train_dataset_len)) - set(initial_seen_indice) - set(unseen_exclude_indices) - set(unseen_indices))
    # seen_indices_from_remained = random.sample(remained_real_inices, len(remained_real_inices) // 2)
    # unseen_indices_from_remained = list(set(remained_real_inices) - set(seen_indices_from_remained))


    # accelerator.print("initial_seen_indices", len(initial_seen_indice))
    # accelerator.print("seen_indices_from_remained", len(seen_indices_from_remained))
    # accelerator.print("unseen_indices_from_remained", len(unseen_indices_from_remained))
    # accelerator.print("unseen_indices", len(unseen_indices))
    # if args.for_guidance:
    #     seen_indices = initial_seen_indice + unseen_indices_from_remained
    #     unseen_indices = unseen_indices + seen_indices_from_remained
    # else:
    #     seen_indices = initial_seen_indice + seen_indices_from_remained
    #     unseen_indices = unseen_indices + unseen_indices_from_remained

    seen_dataset = NewNewCocoStuffDataset(data_path, indices=seen_indices, 
                                            additional_dataset_paths=additional_dataset_paths, transform=transform)
    unseen_dataset = NewNewCocoStuffDataset(data_path, indices=unseen_indices, transform=transform)
    
    seen_dataloader = torch.utils.data.DataLoader(
        seen_dataset, batch_size=args.train_batch_size,
        drop_last=True, num_workers=num_workers
    )

    unseen_dataloader = torch.utils.data.DataLoader(
        unseen_dataset, batch_size=args.train_batch_size,
        drop_last=True, num_workers=num_workers
    )
    

    val_seen_idx = random.sample(range(len(seen_dataset)), int(len(seen_dataset) * 0.1))
    val_seen_dataset = torch.utils.data.Subset(seen_dataset, val_seen_idx)
    val_unseen_idx = random.sample(range(len(unseen_dataset)), int(len(unseen_dataset) * 0.1))
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

    # num_workers = 4
    # if os.environ.get("PJRT_DEVICE", None) == "TPU":
    #     # num_workers = 4 * xm.xrt_world_size()
    #     num_workers = 4
    # elif torch.cuda.is_available():
    #     num_workers = 4 * torch.cuda.device_count()
    
        
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, 
    #     drop_last=True, num_workers=num_workers
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn_concatenate,
    #     drop_last=True, num_workers=num_workers
    # )

    # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # num_update_steps_per_epoch = math.ceil(len(seen_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(min(len(seen_dataloader), len(unseen_dataloader)) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # discriminator, optimizer, train_dataloader = accelerator.prepare(
    #     discriminator, optimizer, train_dataloader
    # )
    discriminator, optimizer, seen_dataloader, unseen_dataloader = accelerator.prepare(
        discriminator, optimizer, seen_dataloader, unseen_dataloader
    )

    
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    dataloader_len = max(len(seen_dataloader), len(unseen_dataloader))

    seen_dataloader = infinite_dataloader(seen_dataloader)
    unseen_dataloader = infinite_dataloader(unseen_dataloader)

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

    vae.to(device=accelerator.device)
    vae.to(dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_local_main_process:
        saved_args = copy.copy(args)
        saved_args.num_bucket_per_side = ' '.join([str(each) for each in saved_args.num_bucket_per_side])
        saved_args.train_text_encoder_params = ' '.join(saved_args.train_text_encoder_params)
        accelerator.init_trackers("text2image-fine-tune", config=vars(saved_args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
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
    loss_function = nn.BCELoss()

    def validataion(val_seen_dataloader, val_unseen_dataloader):
        discriminator.eval()
        seen_loss = 0.0
        unseen_loss = 0.0
        seen_acc = 0.0
        unseen_acc = 0.0
        seen_total = 0
        unseen_total = 0
        val_seen_dataloader = tqdm(val_seen_dataloader, disable=not accelerator.is_local_main_process)
        val_unseen_dataloader = tqdm(val_unseen_dataloader, disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            # iterate separately
            for step, batch in enumerate(val_seen_dataloader):
                pixel_values = batch[0].to(weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                dummy_timesteps = torch.zeros(latents.shape[0], ).to(latents.device)
                disc_pred = discriminator(latents, dummy_timesteps)
                disc_pred = F.sigmoid(disc_pred)
                # disc_target = batch[1]
                disc_target = torch.ones_like(disc_pred)
                disc_pred = disc_pred.float()
                disc_target = disc_target.float()
                loss = loss_function(disc_pred, disc_target)

                if os.environ.get("PJRT_DEVICE", None) == "TPU":
                    xm.mark_step()
                
                seen_loss += loss.detach().item()
                seen_acc += ((disc_pred > 0.5) == disc_target).sum().item()
                seen_total += len(disc_pred)
            seen_loss /= seen_total
            seen_acc /= seen_total

            for step, batch in enumerate(val_unseen_dataloader):
                pixel_values = batch[0].to(weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                dummy_timesteps = torch.zeros(latents.shape[0], ).to(latents.device)
                disc_pred = discriminator(latents, dummy_timesteps)
                disc_pred = F.sigmoid(disc_pred)
                # disc_target = batch[1]
                disc_target = torch.zeros_like(disc_pred)
                disc_pred = disc_pred.float()
                disc_target = disc_target.float()
                loss = loss_function(disc_pred, disc_target)

                if os.environ.get("PJRT_DEVICE", None) == "TPU":
                    xm.mark_step()

                unseen_loss += loss.detach().item()
                unseen_acc += ((disc_pred > 0.5) == disc_target).sum().item()
                unseen_total += len(disc_pred)
            unseen_loss /= unseen_total
            unseen_acc /= unseen_total
        
        # print
        accelerator.print("Validation seen loss: ", seen_loss)
        accelerator.print("Validation unseen loss: ", unseen_loss)
        accelerator.print("Validation seen acc: ", seen_acc)
        accelerator.print("Validation unseen acc: ", unseen_acc)
        
        # Log the results using wandb
        accelerator.log({"val_seen_loss": seen_loss, "val_unseen_loss": unseen_loss, "val_seen_acc": seen_acc, "val_unseen_acc": unseen_acc}, step=global_step)

        # return seen_loss, unseen_loss, seen_acc, unseen_acc


    epoch = 0
    # for epoch in range(args.num_train_epochs):
    while True:
        train_loss = 0.0
        # for step, batch in enumerate(train_dataloader):
        seen_batch = next(seen_dataloader)
        unseen_batch = next(unseen_dataloader)
        # for gradient accumulation
        # with accelerator.accumulate(unet):
        
        with torch.autocast(autocast_device, dtype=weight_dtype):
            # Convert images to latent space
            # images: [B, 3, 512, 512], latents: [B, 4, 64, 64]
            # pixel_values = batch[0].to(weight_dtype)
            seen_pixel_values = seen_batch[0].to(weight_dtype)
            unseen_pixel_values = unseen_batch[0].to(weight_dtype)
            pixel_values = torch.cat([seen_pixel_values, unseen_pixel_values], dim=0)
            # latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = vae.encode(pixel_values).latent_dist.sample()
            # multiply with the scalr factor
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            bsz = latents.shape[0]

            # Predict the noise residual and compute loss
            # noise_pred = unet(noisy_latents, timesteps).sample
            dummy_timesteps = torch.zeros(bsz, ).to(latents.device)
            disc_pred = discriminator(latents, dummy_timesteps)
            disc_pred = F.sigmoid(disc_pred)


        disc_target = torch.zeros_like(disc_pred)
        disc_target[:len(seen_pixel_values)] = 1
        disc_pred = disc_pred.float()
        disc_target = disc_target.float()
        # loss = F.binary_cross_entropy_with_logits(disc_pred, disc_target)
        loss = loss_function(disc_pred, disc_target)
        # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss = loss.mean()
        # Backpropagate
        accelerator.backward(loss)

        # for param in discriminator.parameters():
        #     accelerator.print(param)
        #     break
        # count = 0
        # for name, param in discriminator.named_parameters():
        #     if "time_embed" in name:
        #         continue
        #     accelerator.print(name, torch.mean(param.grad)) # time embed 제외
        #     count += 1
        #     if count > 5:
        #         break

        # params_to_clip = (discriminator.parameters())
        # accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        
        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()

        # Gather the losses across all processes for logging (if we use distributed training).
        # xm.mark_step()
        if os.environ.get("PJRT_DEVICE", None) == "TPU":
            xm.mark_step()
        train_loss += loss.detach().item() / args.gradient_accumulation_steps

        # Checks if the accelerator has performed an optimization step behind the scenes
        progress_bar.update(1)
        global_step += 1

        logs = {
            "step_loss": loss.detach().item(), 
            # "lr": lr_scheduler.get_last_lr()[0],
            "epoch": epoch,
            "step": global_step,
        }

        if autocast_device == "xla":
            tpu_mem_info = xm.get_memory_info(xm.xla_device())
            tpu_mem_used_mb = tpu_mem_info["bytes_used"] // 1024 // 1024
            logs.update({"mem_mb": tpu_mem_used_mb})

        accelerator.log(logs, step=global_step)
        progress_bar.set_postfix(**logs)
        train_loss = 0.0

        if global_step >= args.max_train_steps:
            break

        # save current checkpoint every 500 iterations
        if global_step % args.save_ckpt_freq == 1:
            accelerator.wait_for_everyone()
            accelerator.print("Save start")
            # Save peft
            if accelerator.is_local_main_process:
                discriminator_unwrap = accelerator.unwrap_model(discriminator)
                discriminator_cpu = copy.deepcopy(discriminator_unwrap).to("cpu")
                split_embedding_state_dict = discriminator_cpu.state_dict()

                checkpoint_dir = os.path.join(args.output_dir, 'checkpoint', 'iter_' + str(global_step))
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(split_embedding_state_dict, os.path.join(checkpoint_dir, "discriminator.pth"))

            accelerator.print("save done")
            accelerator.wait_for_everyone()

            # Validate
            # validataion(val_seen_dataloader, val_unseen_dataloader)

    # Create the pipeline using the trained modules and save it.
    # EMA model: not used during training, just record it and save it finally
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        discriminator_unwrap = accelerator.unwrap_model(discriminator)
        discriminator_cpu = copy.deepcopy(discriminator_unwrap).to("cpu")
        split_embedding_state_dict = discriminator_cpu.state_dict()

        # Save split embedding
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoint', 'iter_' + str(global_step))
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(split_embedding_state_dict, os.path.join(checkpoint_dir, "discriminator.pth"))

    if args.push_to_hub:
        repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()

