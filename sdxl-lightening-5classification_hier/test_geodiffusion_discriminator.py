

import os
import numpy as np
from PIL import Image

from argparse import ArgumentParser
import torch

from mmengine.config import Config

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from safetensors import safe_open

from geo_models.embed import SplitEmbedding
from geo_utils.data.missing_person import MissingPersonDataset
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import load_model, measure_bald

# TMP
from geo_utils.guidance_utils import measure_prob_classifier
# from classifier.classifier import create_classifier, classifier_defaults
from geo_models.classifier.classifier import create_classifier, classifier_defaults, load_discriminator

import os
from functools import partial 


import sys
import yaml

########################
# Set random seed
#########################
from accelerate.utils import set_seed
set_seed(0)

########################
# Parsers
#########################
parser = ArgumentParser(description='Generation script')
# parser.add_argument('ckpt_path', type=str)
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--nsamples', type=int, default=4)
parser.add_argument('--cfg_scale', type=float, default=5)
parser.add_argument("--use_dpmsolver", action="store_true")
parser.add_argument('--num_inference_steps', type=int, default=100)
parser.add_argument('--trained_text_encoder', action="store_true")

parser.add_argument('--freedom', action="store_true")
parser.add_argument('--freedom_scale', type=int, default=1)
parser.add_argument('--freedom_model_type', type=str, default="classifier")
parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")
parser.add_argument('--freedom_bald_iteration', type=int, default=10)
# parser.add_argument('--yolo_config', type=str, default='../yolov7/yolo_active_learning_default.yaml')

parser.add_argument('--output_dir', type=str, default='output_img/tmp')

parser.add_argument('--start', type=int, default=70)
parser.add_argument('--end', type=int, default=30)

## LoRA mode
parser.add_argument('--lora_mode', action="store_true", help='LoRA')
parser.add_argument(
  '--lora_model', type=str, 
  default='runwayml/stable-diffusion-v1-5', help='LoRA model name')

# copy from training script
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default="configs/data/missing_person_512x512.py",
    help="The config of the Dataset, leave as None if there's only one config.",
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

args = parser.parse_known_args()[0]

print("{}".format(args).replace(', ', ',\n'))

# Set accelerator
device = "cpu"
if os.environ.get("PJRT_DEVICE", None) == "TPU":
  import torch_xla.core.xla_model as xm
  device = xm.xla_device()
elif torch.cuda.is_available():
  device = "cuda"

########################
# Build pipeline
#########################

vae = AutoencoderKL.from_pretrained("work_dirs/missing_person_512x512_240629/checkpoint/iter_2820", subfolder="vae")
vae = vae.to(device)

# classifier_dir = "work_dirs/missing_person_512_512_240904_clf_2/checkpoint/iter_1620/classifier.pth"
# classifier_dir = "work_dirs/missing_person_512_512_240904_clf_dummy/checkpoint/iter_1620/classifier.pth"

# classifier_dir = "work_dirs/missing_person_512_512_240905_clf/checkpoint/iter_1560/classifier.pth"
# classifier_ckpt = torch.load(classifier_dir)
# classifier_model = create_classifier(**classifier_defaults())
# classifier_model.load_state_dict(classifier_ckpt)
classifier_dir = "work_dirs/missing_person_512_512_240910_disc/checkpoint/iter_2760/discriminator.pth"
classifier_model = load_discriminator(classifier_dir, condition=False, eval=True, channel=4)
classifier_model = classifier_model.to(device)


########################
# Load dataset
# Note: remember to disable randomness in data augmentations !!!!!! (TODO CHECK)
#########################
tokenizer = None
# img_prefix = "/home/djfelrl11/geodiffusion/work_dirs/missing_person_512x512_240629/checkpoint/iter_2820/output_img/bald_clf/"
# img_prefix = "/home/djfelrl11/geodiffusion/work_dirs/missing_person_512x512_240629/checkpoint/iter_2820/output_img/no_bald_clf/"
# img_prefix = "/home/djfelrl11/data/train/real_images/"
# img_prefix = "/home/djfelrl11/data/val/real_images/"
img_prefix = "/home/djfelrl11/geodiffusion/work_dirs/missing_person_512x512_240629/checkpoint/iter_2820/output_img/discriminator_guidance_on/"
num_bucket_per_side = [2, 2]
# if (len(args.num_bucket_per_side) == 1):
#   args.num_bucket_per_side *= 2
dataset_cfg = Config.fromfile(args.dataset_config_name)
dataset_cfg.data.train.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=num_bucket_per_side))
dataset_cfg.data.val.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=num_bucket_per_side))
# dataset_cfg.update(dict(data_root=data_root))
dataset_cfg.data.train.update(dict(img_prefix=img_prefix))
dataset_cfg.data.val.update(dict(img_prefix=img_prefix))
dataset_cfg.data.train.pipeline[3]["flip_ratio"] = 0.0
dataset_cfg.data.val.pipeline[3]["flip_ratio"] = 0.0

width, height = dataset_cfg.data.train.pipeline[2].img_scale if args.split == 'train' else dataset_cfg.data.val.pipeline[2].img_scale
# dataset = build_dataset(dataset_cfg.data.train) if args.split == 'train' else build_dataset(dataset_cfg.data.val)
# dataset = MissingPersonDataset(**dataset_cfg.data.train, tokenizer=tokenizer) \
#   if args.split == 'train' \
#   else MissingPersonDataset(**dataset_cfg.data.val, tokenizer=tokenizer)

dataset = MissingPersonDataset(**dataset_cfg.data.train, tokenizer=tokenizer)
# dataset = MissingPersonDataset(**dataset_cfg.data.val, tokenizer=tokenizer)
print('Image resolution: {} x {}'.format(width, height))
print(len(dataset))


########################
# Set index range
#########################
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

########################
# Generation
#########################
scale = args.cfg_scale
n_samples = args.nsamples

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  # pipe.safety_checker = null_safety

autocast_device = "xla" if os.environ.get("PJRT_DEVICE", None) == "TPU" else "cuda"

result_dict = {}

# with open('bald_array_validation_set_for_clf.yaml', "r") as f:
#   bald_result = yaml.load(f, Loader=yaml.FullLoader)

for i, example in enumerate(dataloader):
  prompt = example["text"]
  print("prompt: ", prompt)
  # run generation

  with torch.autocast(autocast_device):
    pixel_values = example['pixel_values'].to(device)
    # pixel_values = torch.randn_like(pixel_values) * 0.1 + pixel_values
    pixel_values = torch.randn_like(pixel_values).to(device).clip(-1, 1)
    latent = vae.encode(pixel_values).latent_dist.sample()
    latent = latent * 0.18215
    dummy_timestep = torch.zeros(pixel_values.size(0), ).to(pixel_values.device)
    classifier_value = classifier_model(latent, dummy_timestep)
    sigmoid_classifier_value = torch.sigmoid(classifier_value)
  filename = example["img_metadata"]['file_name'][0]
  xm.mark_step()
  result_dict[filename] = sigmoid_classifier_value.cpu().clone().detach().item()
  print(result_dict[filename])

  # save results
with open('predicted_dummy.yaml', "w") as f:
  yaml.dump(result_dict, f)