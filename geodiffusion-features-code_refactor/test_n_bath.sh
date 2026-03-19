#!/bin/bash
# GPU 5 사용
# 공통 인자들 설정
CKPT_PATH="runwayml/stable-diffusion-v1-5"   
NUM_INFERENCE_STEPS=50                       
CFG_SCALE=5                                  
CYCLE=0                                      
TRAINED_TEXT_ENCODER="--trained_text_encoder"  
FREEDOM="--freedom"                          
FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/nudity_classifier_balanced_time_dependent/checkpoint/iter_2801/nudity_classifier.pth"
CONFIG_PATH="./configs/models/time_dependent_discriminator.yaml"
G_Start=1

# 단일 프롬프트 파일 및 nsamples 설정
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_body.txt"
PROMPT_FILEC="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/coco_30.txt"

NSAMPLES=1
MAX_CONCURRENCY=1

# F_SCALE 값을 1부터 6까지 0.5 단위로 변경하면서 작업 실행
for F_SCALE in $(seq 25 5 40); do
    # 출력 디렉토리 생성: F_SCALE 값과 프롬프트 파일 basename 포함
    PROMPT_NAME=$(basename "${PROMPT_FILE}" .txt)
    OUTPUT_DIR="Continual/output_img_Nudity_logodd/${F_SCALE}_${PROMPT_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Running with F_SCALE=${F_SCALE}, PROMPT_FILE=${PROMPT_FILE}, NSAMPLES=${NSAMPLES}"
    
    # nohup으로 백그라운드 실행한 후, PID를 저장하고 wait로 해당 작업 종료까지 대기
    CUDA_VISIBLE_DEVICES=0 nohup python test_nudity.py \
        --ckpt_path "$CKPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --prompt_file "$PROMPT_FILE" \
        --cycle "$CYCLE" \
        --nsamples "$NSAMPLES" \
        --cfg_scale "$CFG_SCALE" \
        $FREEDOM \
        --freedom_scale "$F_SCALE" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --freedom_model_args_file "$CONFIG_PATH" \
        --freedom_model_ckpt "$FREEDOM_MODEL_CKPT" \
        --guide_start "$G_Start" > "${OUTPUT_DIR}/nohup.log" 2>&1 &
    
    PID=$!
    while [ $(jobs -pr | wc -l) -ge $MAX_CONCURRENCY ]; do
        wait -n
    done

    # # 출력 디렉토리 생성: F_SCALE 값과 프롬프트 파일 basename 포함
    # PROMPT_NAME=$(basename "${PROMPT_FILEC}" .txt)
    # OUTPUT_DIR="Continual/output_img_Nudity/discriminator_${F_SCALE}_${PROMPT_NAME}"
    # mkdir -p "$OUTPUT_DIR"
    
    # echo "Running with F_SCALE=${F_SCALE}, PROMPT_FILE=${PROMPT_FILEC}, NSAMPLES=${NSAMPLES}"
    
    # # nohup으로 백그라운드 실행한 후, PID를 저장하고 wait로 해당 작업 종료까지 대기
    # CUDA_VISIBLE_DEVICES=0 nohup python test_nudity.py \
    #     --ckpt_path "$CKPT_PATH" \
    #     --output_dir "$OUTPUT_DIR" \
    #     --prompt_file "$PROMPT_FILEC" \
    #     --cycle "$CYCLE" \
    #     --nsamples "$NSAMPLES" \
    #     --cfg_scale "$CFG_SCALE" \
    #     $FREEDOM \
    #     --freedom_scale "$F_SCALE" \
    #     --num_inference_steps "$NUM_INFERENCE_STEPS" \
    #     --freedom_model_args_file "$CONFIG_PATH" \
    #     --freedom_model_ckpt "$FREEDOM_MODEL_CKPT" \
    #     --guide_start "$G_Start" > "${OUTPUT_DIR}/nohup.log" 2>&1 &
    
    # PID=$!
    # while [ $(jobs -pr | wc -l) -ge $MAX_CONCURRENCY ]; do
    #     wait -n
    # done

done
