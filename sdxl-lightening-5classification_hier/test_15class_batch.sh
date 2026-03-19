#!/usr/bin/env bash

#───────────────────────────────────────────────────────────────────────────────#
# 연속적으로 여러 F_SCALE 값을 적용하여 test_11class.py 실행 (순차 실행)
#───────────────────────────────────────────────────────────────────────────────#

export CUDA_VISIBLE_DEVICES=5

# (1) 공통 인자
CKPT_PATH="runwayml/stable-diffusion-v1-5"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_naked_body.txt"
NSAMPLES=1
CFG_SCALE=5
NUM_INFERENCE_STEPS=50
FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/unlearning/15_classificaiton/work_dirs/fifteen_class_output/checkpoint/step_41600/classifier.pth"
CONFIG_PATH="/mnt/home/yhgil99/unlearning/15_classificaiton/configs/models/time_dependent_discriminator15.yaml"
G_START=1
GUIDING_CLASS=1

# (2) F_SCALE 값 리스트
for F_SCALE in 5 10 15 20 25; do
    # (3) 출력 디렉토리
    OUTPUT_DIR="Continual/CountryNudeBodyMulti1,10/${F_SCALE}"
    mkdir -p "$OUTPUT_DIR"

    echo "▶ Running test_11class.py with freedom_scale=${F_SCALE}"
    date

    # (4) test_11class.py 호출 (동기 실행)
    python test_11class.py \
        --ckpt_path             "$CKPT_PATH" \
        --output_dir            "$OUTPUT_DIR" \
        --prompt_file           "$PROMPT_FILE" \
        --nsamples              "$NSAMPLES" \
        --cfg_scale             "$CFG_SCALE" \
        --num_inference_steps   "$NUM_INFERENCE_STEPS" \
        --freedom               \
        --freedom_scale         "$F_SCALE" \
        --freedom_model_args_file "$CONFIG_PATH" \
        --freedom_model_ckpt    "$FREEDOM_MODEL_CKPT" \
        --guide_start           "$G_START" \
        --guiding_class         "$GUIDING_CLASS"

    # (5) 한 번 실행이 끝나면 로그 남기기
    echo "✔ Finished F_SCALE=${F_SCALE}"
    date
    echo "──────────────────────────────────────────────────"
done

echo "✅ All runs completed."
