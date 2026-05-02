

import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from multiprocessing import Pool

import torch
from accelerate import Accelerator

from mmengine.config import Config

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from safetensors import safe_open

from geo_models.embed import SplitEmbedding
from geo_utils.data.missing_person import MissingPersonDataset
from geo_utils.data.new_coco_stuff import NewCOCOStuffDataset, NewCOCOStuffSubsetDataset
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline, CustomStableDiffusionImg2ImgPipeline
from geo_utils.guidance_utils import GuidanceModel

from tqdm import tqdm


import os
from functools import partial 
from multiprocessing import Pool


import sys
import yaml
import random
# sys.path.append("/home/djfelrl11/yolov7")
# import load_model

########################
# Set random seed
#########################
# from accelerate.utils import set_seed
# set_seed(0)

########################
# Parsers
#########################
def parse_args():
  parser = ArgumentParser(description='Generation script')
  parser.add_argument('ckpt_path', type=str)
  parser.add_argument('--split', type=str, default='val')
  parser.add_argument('--nsamples', type=int, default=4)
  parser.add_argument('--cfg_scale', type=float, default=5)
  parser.add_argument("--use_dpmsolver", action="store_true")
  # parser.add_argument('--num_inference_steps', type=int, default=100)
  parser.add_argument('--num_inference_steps', type=int, default=50)
  parser.add_argument('--trained_text_encoder', action="store_true")

  parser.add_argument('--freedom', action="store_true")
  parser.add_argument('--freedom_scale', type=float, default=1.0)
  parser.add_argument('--freedom_model_type', type=str, default="classifier")
  parser.add_argument('--freedom_model_args_file', type=str, default="configs/models/classifier.yaml")
  parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")
  parser.add_argument('--freedom_bald_iteration', type=int, default=10)
  # parser.add_argument('--yolo_config', type=str, default='../yolov7/yolo_active_learning_default.yaml')

  parser.add_argument('--output_dir', type=str, default='output_img/tmp')

  ## LoRA mode
  parser.add_argument('--lora_mode', action="store_true", help='LoRA')
  parser.add_argument(
    '--lora_model', type=str, 
    default='runwayml/stable-diffusion-v1-5', help='LoRA model name')

  # copy from training script
  parser.add_argument(
      "--dataset_config_name",
      type=str,
      default=None,
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

  parser.add_argument(
    "--distributed_inference",
    action="store_true",
    help="Whether to use distributed inference",
  )

  parser.add_argument(
    "--generated_bbox_dir",
    type=str,
    default=None,
    help="The directory to save the generated bounding boxes",
  )

  parser.add_argument(
    "--cycle",
    type=int,
    default=None,
    help="The cycle number",
  )

  parser.add_argument(
    "--manual_generation_file_list",
    type=str,
    default=None,
    help="The file list for manual generation",
  )

  parser.add_argument(
    "--guide_start",
    type=int,
    default=0,
    help="The start index for guidance",
  )

  parser.add_argument(
    "--img2img",
    action="store_true",
    help="Whether to use img2img pipeline",
  )

  parser.add_argument(
    "--strength",
    type=float,
    help="The strength of the img2img guidance",
  )

  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="The seed for random number generator",
  )

  parser.add_argument(
    "--grad_only_bbox",
    action="store_true",
    help="Whether to use only bbox gradient",
  )

  parser.add_argument(
    "--grad_bbox_reverse",
    action="store_true",
    help="Whether to reverse the bbox gradient",
  )

  parser.add_argument(
    "--geodiffusion_scale",
    type=float,
    default=4.0,
    help="The scale for geodiffusion",
  )

  args = parser.parse_known_args()[0]

  print("{}".format(args).replace(', ', ',\n'))
  return args

def save_image(image, img_metadata, bounding_box, root="work_dirs/missing_person_512x512_240629/checkpoint/iter_2820", check_bbox=False, is_real=False):
  path = img_metadata['file_name']
  img_height = img_metadata['height']
  img_width = img_metadata['width']

  image = np.asarray(image)

  num_of_bbox = len(bounding_box)

  if check_bbox:
    image = image.copy()
    bboxes = img_metadata['bbox']
    for i, box in enumerate(bboxes):
      x, y, w, h = box
      x1, y1, x2, y2 = x, y, x + w, y + h
      x2 = min(x2, img_width - 1)
      y2 = min(y2, img_height - 1)
      x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
      image[y1:y2, x1] = 255
      image[y1:y2, x2] = 255
      image[y1, x1:x2] = 255
      image[y2, x1:x2] = 255

  image = Image.fromarray(image, mode='RGB')
  image = image.resize((img_width, img_height))
  # path = os.path.join(root, path[:-4] + '.jpg') if is_real else os.path.join(root, path[:-4] + '_fake.jpg')

  path = os.path.join(root, path[:-4] + '.jpg')
  image.save(path)

def collate_fn(examples):
  pixel_values = torch.stack([example["pixel_values"] for example in examples]).float()
  input_ids = torch.stack([example["input_ids"] for example in examples]).to(memory_format=torch.contiguous_format).long()
  metadata = [example["img_metadata"] for example in examples]
  text = [example["text"] for example in examples]
  instance_text = [example["instance_text"] for example in examples]
  bbox = [example["bbox"] for example in examples]
  bbox_binary_mask = [example.get("bbox_binary_mask", None) for example in examples]
  if bbox_binary_mask[0] is None:
    bbox_binary_mask = None
  else:
    bbox_binary_mask = torch.stack(bbox_binary_mask).to(memory_format=torch.contiguous_format).float()
    bbox_binary_mask = torch.unsqueeze(bbox_binary_mask, 1)
  
  return {
      "pixel_values": pixel_values,
      "text": text,
      "instance_text": instance_text,
      "input_ids": input_ids,
      "img_metadata": metadata,
      "bboxes": bbox,
      "bbox_binary_mask": bbox_binary_mask,
  }


# def inference_main(model=None):
def main(model=None):

  if os.environ.get("PJRT_DEVICE", None) == "TPU":
    import torch_xla.core.xla_model as xm
    import torch_xla
    import torch_xla.runtime as xr
  args = parse_args()

  # Set accelerator
  accelerator = Accelerator()
  device = accelerator.device

  ########################
  # Build pipeline
  #########################
  ckpt_path = args.lora_model if args.lora_mode else args.ckpt_path 
  tokenizer = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
  vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae")
  unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
  text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder")

  if args.trained_text_encoder:
    if args.lora_mode:
      original_num_words = len(tokenizer.get_vocab())
      if args.num_bucket_per_side is not None:
          if len(args.num_bucket_per_side) == 1:
              args.num_bucket_per_side *= 2
          new_tokens = ["<l{}>".format(i) for i in range(int(args.num_bucket_per_side[0] * args.num_bucket_per_side[1]))]
          tokenizer.add_tokens(new_tokens)
      text_encoder.resize_token_embeddings(len(tokenizer.get_vocab()))
      text_encoder.text_model.embeddings.token_embedding = SplitEmbedding(text_encoder.text_model.embeddings.token_embedding, len(tokenizer.get_vocab()))
      text_encoder.load_state_dict(torch.load(f"{args.ckpt_path}/text_encoder/model.pth"))
    else:
      num_words = text_encoder.get_input_embeddings().num_embeddings
      text_encoder.text_model.embeddings.token_embedding = SplitEmbedding(text_encoder.text_model.embeddings.token_embedding, num_words)
      # text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder")
      with safe_open(f"{ckpt_path}/text_encoder/model.safetensors", framework="pt") as f:
        original_state_dict = text_encoder.state_dict()
        for key in f.keys():
          if "token_embedding" in key:
            original_state_dict[key] = f.get_tensor(key)
        text_encoder.load_state_dict(original_state_dict)
  # breakpoint()
  # Load the model
  if args.img2img:
    pipeline_class = CustomStableDiffusionImg2ImgPipeline
  else:
    pipeline_class = CustomStableDiffusionPipeline
    
  pipe = pipeline_class(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=PNDMScheduler.from_config(ckpt_path, subfolder="scheduler"),
    safety_checker=None,
    feature_extractor=CLIPFeatureExtractor.from_pretrained(ckpt_path, subfolder="feature_extractor"),
  )

  if args.lora_mode:
    pipe.load_lora_weights(os.path.join(args.ckpt_path, "lora"))

  # if args.geodiffusion_mode == "coco_stuff":
  if "coco_stuff" in args.dataset_config_name: # TODO: Maybe it should be needed to all of the dataset
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


  pipe = pipe.to(device)
  print("pipe device: ", pipe.device)


  ########################
  # Load dataset
  # Note: remember to disable randomness in data augmentations !!!!!! (TODO CHECK)
  #########################
  if (len(args.num_bucket_per_side) == 1):
    args.num_bucket_per_side *= 2
  dataset_cfg = Config.fromfile(args.dataset_config_name)
  dataset_cfg.data.train.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=args.num_bucket_per_side))
  dataset_cfg.data.val.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=args.num_bucket_per_side))
  dataset_cfg.data.train.pipeline[3]["flip_ratio"] = 0.0
  dataset_cfg.data.val.pipeline[3]["flip_ratio"] = 0.0
  if args.generated_bbox_dir is not None:
    dataset_cfg.data.val.update(dict(ann_file=args.generated_bbox_dir))
    dataset_cfg.data.train.update(dict(ann_file=args.generated_bbox_dir))

  width, height = dataset_cfg.data.train.pipeline[2].img_scale if args.split == 'train' else dataset_cfg.data.val.pipeline[2].img_scale
  
  # dataset_cfg.data.train.ann_file = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/coco-stuff-instance/instances_stuff_train2017.json"
  # dataset_cfg.data.train.ann_file = "../coco_2017/annotations/instances_train2017.json"
  # dataset_cfg.data.train.ann_file = "../coco_2017/annotations/instances_stuff_train2017.json" # default
  dataset_cfg.data.train.ann_file = "/mnt/disk/coco_2017/annotations/instances_stuff_train2017.json"

  if args.manual_generation_file_list is not None:
    subset_dict = yaml.load(open(args.manual_generation_file_list), Loader=yaml.FullLoader)
    subset_list = subset_dict.keys()
    dataset = NewCOCOStuffSubsetDataset(**dataset_cfg.data.train, tokenizer=tokenizer, only_generation=True, subset_list=subset_list)

  # dataset = NewCOCOStuffDataset(**dataset_cfg.data.train, tokenizer=tokenizer, only_generation=True)
  # print('Image resolution: {} x {}'.format(width, height))
  print("Dataset length", len(dataset))


  # Add yolo to calculate active learning metric
  if args.freedom:
    freedom_model = GuidanceModel(pipe, args.freedom_model_args_file, args.freedom_model_ckpt, pipe.device)
    tmp_diff_value_list = []

    def pre_freedom_iteration(diffusion_pipeline, step, timestep, callback_kwargs, freedom_model, freedom_scale, guide_start=0):
      # step: increasing: 1 -> 100
      # timestep: decreasing: 1000 -> 1
      if guide_start <= step:
        # return freedom_model.guidance(diffusion_pipeline, callback_kwargs, step, timestep, freedom_scale)
        guidance_result = freedom_model.guidance(diffusion_pipeline, callback_kwargs, step, timestep, freedom_scale)
        tmp_diff_value_list.append(guidance_result["differentiate_value"])
        return guidance_result
      else:
        return callback_kwargs

    freedom_iteration = partial(pre_freedom_iteration, 
      freedom_model=freedom_model, freedom_scale=args.freedom_scale, guide_start=args.guide_start)
    # freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]
    freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents", "bbox_binary_mask"]
    freedom_callback_on_step_end_tensor_input += ["instance_prompt_embeds"] if args.freedom_model_type == "augmented_discriminator" else []


  ########################
  # Set index range
  #########################
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=4)
  dataloader = accelerator.prepare(dataloader)
  progress_bar = tqdm(dataloader, desc="Generating images", 
                      total=len(dataloader)*accelerator.num_processes, 
                      disable=not accelerator.is_local_main_process)
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
    pipe.safety_checker = null_safety

  autocast_device = "xla" if os.environ.get("PJRT_DEVICE", None) == "TPU" else "cuda"
  pool = Pool(4)

  guidance_evaluation = {}
  for i, example in enumerate(dataloader): # ~800
    prompt = example["text"]
    input_images = example["pixel_values"]
    bounding_box = example["bboxes"]
    bbox_binary_mask = example.get("bbox_binary_mask", None)
    if args.grad_only_bbox:
      assert bbox_binary_mask is not None
      bbox_binary_mask = bbox_binary_mask.to(device)
      bbox_binary_mask = bbox_binary_mask.requires_grad_(False)
    if args.grad_bbox_reverse:
      assert args.grad_only_bbox
      bbox_binary_mask = 1 - bbox_binary_mask
    torch.manual_seed(i + accelerator.num_processes * accelerator.process_index + args.seed * 100000)
    if os.environ.get("PJRT_DEVICE", None) == "TPU":
      xm.set_rng_state(i + accelerator.num_processes * accelerator.process_index + args.seed * 100000)

    instance_prompt = example["instance_text"] if args.freedom_model_type == "augmented_discriminator" else None
    if instance_prompt is not None:
      instance_prompt = torch.tensor(pipe.tokenizer(instance_prompt, max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True).input_ids)
      instance_prompt = pipe.text_encoder(instance_prompt.to(device))[0]
      instance_prompt = instance_prompt.detach().requires_grad_(False)


    input_dict = {
      "prompt": prompt,
      "instance_prompt_embeds": instance_prompt,
      "guidance_scale": scale, 
      "num_inference_steps": args.num_inference_steps, 
      "height": int(height), 
      "width": int(width), 
      "callback_on_step_end": freedom_iteration if args.freedom else None, 
      "callback_on_step_end_tensor_inputs": freedom_callback_on_step_end_tensor_input if args.freedom else None,
      "callback": lambda *args: xm.mark_step() if autocast_device == "xla" else None, 
      "callback_steps": 1,
      # "guidance_scale": 4.0 if args.geodiffusion_mode == "coco_stuff" else 7.5,
      "guidance_scale": args.geodiffusion_scale,
      "bbox_binary_mask": bbox_binary_mask if args.grad_only_bbox else None,
    }

    if args.img2img:
      input_dict["image"] = input_images
      input_dict["strength"] = args.strength

    if args.freedom:
      images = pipe(**input_dict).images
    else:
      images = pipe(
        prompt, 
        guidance_scale=scale, 
        num_inference_steps=args.num_inference_steps, 
        height=int(height), width=int(width), 
        callback=lambda *args: xm.mark_step() if autocast_device == "xla" else None, 
        callback_steps=1).images
    # make directory
    dpm_flag = "_dpmsolver" if args.use_dpmsolver else ""
    # root = os.path.join(ckpt_path, args.output_dir)
    root = os.path.join(args.output_dir, str(args.cycle))

    img_metadata_list = example["img_metadata"]
    
    os.makedirs(root, exist_ok=True)

    input_images = input_images.cpu()

    save_image_partial = partial(save_image, root=root)
    pool.starmap(save_image_partial, zip(images, img_metadata_list, bounding_box))

    tmp_guidance_evaluation_dict = {}
    for img_metadata in img_metadata_list:
      tmp_guidance_evaluation_dict[img_metadata["file_name"]] = []
    
    for diff_value in tmp_diff_value_list:
      assert len(diff_value) == len(img_metadata_list)
      for tmp_guidance_evaluation_key, diff_value_elem in zip(tmp_guidance_evaluation_dict.keys(), diff_value):
        # tmp_guidance_evaluation_dict[tmp_guidance_evaluation_key].append(diff_value_elem.tolist())
        if len(diff_value_elem) == 2:
          diff_value_elem = diff_value_elem[0]

        tmp_guidance_evaluation_dict[tmp_guidance_evaluation_key].append(diff_value_elem.item())
    
    tmp_diff_value_list = []
    guidance_evaluation.update(tmp_guidance_evaluation_dict)

    # with open(os.path.join(root, "guidance_evaluation.yaml"), "w") as f:
    # if accelerator.is_local_main_process:
    #   with open(f"250124guidance_evaluation_with_guidance_{args.freedom_scale}.yaml", "w") as f:
    #     yaml.dump(guidance_evaluation, f)
    
  pool.close()
    

if __name__=="__main__":
  main()