import os, torch, numpy as np
from argparse import ArgumentParser
from PIL import Image
from functools import partial
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import GuidanceModel

def parse_args():
    p = ArgumentParser()
    p.add_argument("ckpt_path")
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--cls_scale", type=float, default=1.0)
    p.add_argument("--clip_scale", type=float, default=0.1)
    p.add_argument('--freedom_scale', type=float, default=10.0)
    p.add_argument("--switch_step", type=int, default=25)
    p.add_argument("--output_dir", default="output_img")
    return p.parse_args()

def save_image(img, fn, size=(512,512)):
    im = Image.fromarray(np.asarray(img)).resize(size)
    im.save(fn)

def main():
    args = parse_args()
    acc = Accelerator()
    device = acc.device

    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = pipe.to(device)

    prompts = [l.strip() for l in open(args.prompt_file)]

    if args.mixed:
        # 3-class classifier GuidanceModel
        cls_gm = GuidanceModel(
            pipe,
            model_config_file="configs/models/classifier.yaml",
            model_ckpt_path="classifier_ckpt/classifier.pth",
            target_class=1,
            device=device
        )
        # CLIP GuidanceModel
        clip_gm = GuidanceModel(
            pipe,
            model_config_file="/configs/models/time_dependent_discriminator.yaml",       # we pass args directly
            model_ckpt_path=None,
            target_class=None,
            device=device
        )
        # override its model_args to use clip
        clip_gm.gradient_model = clip_gm.load_gradient_model(
            "clip",
            {"text":"a fully clothed person","clip_model":"openai/clip-vit-large-patch14"}
        )

        def mixed_cb(pipe, step, ts, cbkw, cls_gm, clip_gm, cls_s, clip_s, sw):
            prev = cbkw["latents"].detach().clone().requires_grad_(True)
            noise = cbkw["noise_pred"]
            if step < sw:
                out = cls_gm.guidance(pipe, cbkw, step, ts, cls_s)
            else:
                out = clip_gm.guidance(pipe, cbkw, step, ts, clip_s)
            return out

        cb = partial(
            mixed_cb,
            cls_gm=cls_gm,
            clip_gm=clip_gm,
            cls_s=args.freedom_scale,
            clip_s=args.freedom_scale,
            sw=args.switch_step
        )
        cb_inputs = ["latents","noise_pred"]
    else:
        cb, cb_inputs = None, None

    os.makedirs(args.output_dir, exist_ok=True)

    for i,prompt in enumerate(prompts):
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        pipe.scheduler._step_index = 0

        out = pipe(
            prompt,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            height=512, width=512,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=cb_inputs,
            callback_steps=1,
            num_images_per_prompt=args.nsamples
        )
        for j, img in enumerate(out.images):
            fn = os.path.join(args.output_dir, f"{i+1}_{j+1}.png")
            save_image(img, fn)

if __name__=="__main__":
    main()
