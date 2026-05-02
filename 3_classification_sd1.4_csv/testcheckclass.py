import os
import csv
import random
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline  # Stable Diffusion 파이프라인 사용
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline, CustomStableDiffusionImg2ImgPipeline

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from geo_utils.guidance_utils import GuidanceModel
from geo_utils.sae_probe import SAEProbe
import numpy as np

########################
# Parsers
#########################
def parse_args():
    parser = ArgumentParser(description='Generation script')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--trained_text_encoder', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cycle', type=int, default=0)

    # Guidance 관련 인자
    parser.add_argument('--freedom', action="store_true", help="Use guidance with trained discriminator")
    parser.add_argument('--freedom_scale', type=float, default=10.0)
    parser.add_argument('--freedom_model_type', type=str, default="classifier")
    parser.add_argument('--freedom_model_args_file', type=str, default="configs/models/classifier.yaml")
    parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")
    parser.add_argument("--guide_start", type=int, default=1, help="The start index for guidance")

    parser.add_argument('--output_dir', type=str, default='output_img/tmp')

    # Prompt file argument
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the file containing prompts")

    # === SAE probe 옵션 ===
    parser.add_argument('--sae_probe', action="store_true",
                        help="Enable SAE feature logging with SAeUron")
    parser.add_argument('--sae_repo', type=str, default="bcywinski/SAeUron",
                        help="HF Hub repo to load SAE from")
    parser.add_argument('--sae_hookpoint', type=str, default="unet.up_blocks.1.attentions.1",
                        help="UNet hookpoint for SAE features")
    parser.add_argument('--sae_topk', type=int, default=32,
                        help="Top-K SAE features to average for FAI score")
    parser.add_argument('--sae_csv', type=str, default=None,
                        help="Path to save per-step FAI CSV (e.g., Continual/3classifier/10/fai_log.csv)")
    parser.add_argument('--sae_calibrate', action="store_true",
                        help="Run quick calibration to select feature set Fc")
    parser.add_argument('--sae_nudity_prompts', type=str, default=None,
                        help="File with nudity-anchor prompts (one per line)")
    parser.add_argument('--sae_neutral_prompts', type=str, default=None,
                        help="File with neutral prompts (one per line); default: first 10 from --prompt_file")

    # === Classifier dump 옵션 ===
    parser.add_argument('--cls_dump_csv', type=str, default=None,
                        help="Per-step classifier outputs CSV path (e.g., logs/cls_dump.csv)")
    parser.add_argument('--cls_dump_max_step', type=int, default=5,
                        help="Dump steps up to this index (inclusive).")
    parser.add_argument('--cls_dump_after_guidance', action='store_true',
                        help="Dump classifier outputs AFTER guidance update.")

    args = parser.parse_known_args()[0]
    return args

########################
# Save Image Function
########################
def save_image(image, img_metadata, root="output_img"):
    path = img_metadata['file_name']
    img_height = img_metadata['height']
    img_width = img_metadata['width']

    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((img_width, img_height))
    path = os.path.join(root, path[:-4] + '.png')
    image.save(path)

########################
# Helper: dump classifier outputs per step
########################
def dump_classifier_outputs(freedom_model, diffusion_pipeline, kw,
                            timestep, step, prompt_idx, csv_path):
    """
    kw: prev_latents, noise_pred, latents(선택) 를 담은 dict
        (after-guidance일 땐 callback_kwargs와 result_kwargs를 merge해서 넘길 것)
    """
    try:
        if freedom_model is None or csv_path is None:
            return

        # gradient model 핸들
        gm = (freedom_model.get_gradient_model()
              if hasattr(freedom_model, 'get_gradient_model')
              else freedom_model.gradient_model)

        # prev_latents 우선, 없으면 latents로 폴백
        prev_latents = kw.get("prev_latents", None)
        noise_pred   = kw.get("noise_pred", None)
        if prev_latents is None:
            prev_latents = kw.get("latents", None)
            print("[CLS-DUMP] WARN: 'prev_latents' missing; falling back to 'latents' for dump.")
        if prev_latents is None:
            raise KeyError("'prev_latents' and 'latents' are both missing in kwargs for dump.")

        # guidance와 동일 스케일 입력
        img_in = gm.get_scaled_input(diffusion_pipeline, prev_latents, noise_pred, timestep)  # [B,C,H,W]

        # timestep 정규화 (guidance와 동일)
        num_t = diffusion_pipeline.scheduler.num_train_timesteps
        if torch.is_tensor(timestep):
            t = timestep / num_t
            if t.ndim == 0:
                t = t[None]
        else:
            t = torch.tensor([float(timestep) / float(num_t)], device=img_in.device)
        if t.shape[0] != img_in.shape[0]:
            t = t.repeat(img_in.shape[0])

        # logits / probs
        clf = gm.model if hasattr(gm, "model") else gm
        with torch.no_grad():
            logits = clf(img_in, t)            # [B,3]
            probs  = torch.softmax(logits, -1) # [B,3]
            pred   = probs.argmax(-1)          # [B]

        # CSV 저장
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        need_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(['prompt_idx','sample_idx','step','timestep',
                            'logit_0','logit_1','logit_2',
                            'prob_0','prob_1','prob_2','pred_class'])
            ts = float(timestep.detach().item()) if torch.is_tensor(timestep) else float(timestep)
            B = logits.shape[0]
            for b in range(B):
                w.writerow([
                    int(prompt_idx), int(b), int(step), ts,
                    float(logits[b,0].item()), float(logits[b,1].item()), float(logits[b,2].item()),
                    float(probs[b,0].item()),  float(probs[b,1].item()),  float(probs[b,2].item()),
                    int(pred[b].item())
                ])
    except Exception as e:
        print(f"[CLS-DUMP] error: {e}")

########################
# Main Function
########################
def main(model=None):
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # 재현성
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ########################
    # Build Stable Diffusion pipeline (Disable Safety Checker)
    ########################
    pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = pipe.to(device)
    print("Pipe device:", pipe.device, flush=True)

    ########################
    # Load prompt file
    ########################
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}", flush=True)

    ########################
    # SAEProbe 준비
    ########################
    sae_probe = None
    if args.sae_probe:
        sae_probe = SAEProbe(
            pipe=pipe,
            sae_repo=args.sae_repo,
            hookpoint=args.sae_hookpoint,
            device=str(device),
            topk_select=args.sae_topk,
            csv_path=args.sae_csv,
        )

    # (옵션) SAE feature 캘리브레이션
    if sae_probe is not None and args.sae_calibrate:
        if args.sae_nudity_prompts is None:
            nudity_prompts = ["nude human figure", "unclothed human figure"]
        else:
            with open(args.sae_nudity_prompts, "r") as f:
                nudity_prompts = [l.strip() for l in f if l.strip()]

        if args.sae_neutral_prompts is None:
            neutral_prompts = prompts[:10] if len(prompts) >= 10 else prompts
        else:
            with open(args.sae_neutral_prompts, "r") as f:
                neutral_prompts = [l.strip() for l in f if l.strip()]

        base_kws = dict(
            guidance_scale=1.0,
            height=512, width=512,
            num_inference_steps=20,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=None,
            num_images_per_prompt=1,
        )
        sae_probe.calibrate_fc(nudity_prompts, neutral_prompts, base_kws)

    ########################
    # Setup Guidance (trained discriminator)
    ########################
    freedom_model = None
    cb_tensor_inputs_base = None
    if args.freedom:
        freedom_model = GuidanceModel(pipe, args.freedom_model_args_file, args.freedom_model_ckpt, 1, pipe.device)
        cb_tensor_inputs_base = ["latents", "noise_pred", "prev_latents"]
        if args.freedom_model_type == "augmented_discriminator":
            cb_tensor_inputs_base += ["instance_prompt_embeds"]
        print("guidance!!", flush=True)

    ########################
    # Generation Loop
    ########################
    scale = args.cfg_scale
    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    # 콜백: SAE 로깅 + (옵션) guidance + (옵션) classifier dump
    def pre_freedom_iteration(diffusion_pipeline, step, timestep, callback_kwargs,
                              freedom_model, freedom_scale, guide_start=0,
                              prompt_idx=None, sae_probe=None,
                              cls_dump_csv=None, cls_dump_max_step=5,
                              cls_dump_after_guidance=True):
        # 디버그: 첫 스텝에 키 한번 출력
        if step == 0:
            try:
                print("[DEBUG] callback_kwargs keys:", sorted(list(callback_kwargs.keys())), flush=True)
            except Exception:
                pass

        # 1) SAE per-step 로깅
        if sae_probe is not None:
            try:
                sae_probe.log_step(prompt_idx if prompt_idx is not None else -1, step, timestep)
            except Exception as e:
                print(f"[SAEProbe] log_step error: {e}", flush=True)

        # 2) guidance 적용
        result_kwargs = callback_kwargs
        if (freedom_model is not None) and (guide_start <= step):
            result_kwargs = freedom_model.guidance(
                diffusion_pipeline, callback_kwargs, step, timestep,
                freedom_scale, target_class=1
            )

        # 3) classifier 출력 CSV 덤프
        if (cls_dump_csv is not None) and (freedom_model is not None) and (step <= cls_dump_max_step):
            try:
                if cls_dump_after_guidance:
                    # after-guidance: 업데이트된 latents(result) + 원래 prev_latents/noise_pred(callback)를 merge
                    merged = dict(callback_kwargs)
                    merged.update(result_kwargs)  # latents는 업데이트된 것으로 덮어씀
                    src_kwargs = merged
                else:
                    # before-guidance: 원본 그대로
                    src_kwargs = callback_kwargs

                dump_classifier_outputs(freedom_model, diffusion_pipeline, src_kwargs,
                                        timestep, step, prompt_idx, cls_dump_csv)
            except Exception as e:
                print(f"[CLS-DUMP] dump failed at step {step}: {e}", flush=True)

        return result_kwargs

    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt {idx + 1}: {prompt}", flush=True)

        # 프롬프트별 콜백 partial
        if args.freedom:
            freedom_iteration = partial(
                pre_freedom_iteration,
                freedom_model=freedom_model,
                freedom_scale=args.freedom_scale,
                guide_start=args.guide_start,
                prompt_idx=idx,
                sae_probe=sae_probe,
                cls_dump_csv=args.cls_dump_csv,
                cls_dump_max_step=args.cls_dump_max_step,
                cls_dump_after_guidance=bool(args.cls_dump_after_guidance)  # << 고정
            )
            freedom_callback_on_step_end_tensor_input = cb_tensor_inputs_base
        elif sae_probe is not None:
            # guidance OFF + SAE 로깅만
            freedom_iteration = partial(
                pre_freedom_iteration,
                freedom_model=None,
                freedom_scale=0.0,
                guide_start=1_000_000,  # 절대 안 걸리도록
                prompt_idx=idx,
                sae_probe=sae_probe,
                cls_dump_csv=None,  # freedom_model이 없으므로 덤프 비활성
                cls_dump_max_step=args.cls_dump_max_step,
                cls_dump_after_guidance=True
            )
            freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]
        else:
            freedom_iteration = None
            freedom_callback_on_step_end_tensor_input = None

        # Generate image input dictionary
        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512,
            "width": 512,
            "callback_on_step_end": freedom_iteration,
            "callback_on_step_end_tensor_inputs": freedom_callback_on_step_end_tensor_input,
            "callback": None,
            "callback_steps": 1,
            "bbox_binary_mask": None,
            "num_images_per_prompt": args.nsamples,
        }

        with torch.enable_grad():
            generated_images = pipe(**input_dict).images

        # Save first image
        img_metadata = {"file_name": f"{idx + 1}.png", "height": 512, "width": 512}
        save_image(generated_images[0], img_metadata, root=root)

    # === SAE CSV 저장 & 훅 해제
    if sae_probe is not None:
        try:
            sae_probe.flush_csv()
            print("[SAEProbe] CSV saved:", args.sae_csv, flush=True)
        finally:
            sae_probe.close()

if __name__ == "__main__":
    main()
