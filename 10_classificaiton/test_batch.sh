#!/bin/bash
# 두 GPU를 사용할 수 있도록 설정 (여기서는 GPU 4만 사용)
# FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/vangoh_classifier_checkpoint/checkpoint/iter_4001/nudity_classifier.pth"  # 자유 모델 체크포인트
# FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/monet_classifier_chekpoint/checkpoint/iter_4801/nudity_classifier.pth"  # 자유 모델 체크포인트

export CUDA_VISIBLE_DEVICES=0

# 공통 인자들 설정
CKPT_PATH="runwayml/stable-diffusion-v1-5"   
NUM_INFERENCE_STEPS=50                       
CFG_SCALE=5                                  
CYCLE=0                                      
TRAINED_TEXT_ENCODER="--trained_text_encoder"  
FREEDOM="--freedom"                          
FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/monet_classifier_chekpoint/checkpoint/iter_4801/nudity_classifier.pth"  # 자유 모델 체크포인트
CONFIG_PATH="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/configs/models/discriminator.yaml"
G_Start=1
MAX_CONCURRENCY=1

# 프롬프트 파일과 각각의 nsamples 값 설정
prompt_files=(
    "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/artists/Benign_artists.txt"
    "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/artists/ClaudeMonet.txt"
    "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/artists/VincentVangogh.txt"
    "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_body.txt"
    "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/coco_30.txt"
)
nsamples_array=(1 3 3 1 1)

# 각 프롬프트 파일에 대해 반복
for i in "${!prompt_files[@]}"; do
    PROMPT_FILE="${prompt_files[$i]}"
    NSAMPLES="${nsamples_array[$i]}"
    
    # F_SCALE 값을 1부터 6까지 0.5 단위로 변화시키면서 실행
    for F_SCALE in $(seq 1 2 8); do
         # 출력 디렉토리 생성: F_SCALE 값과 프롬프트 파일 이름(basename) 포함
         PROMPT_NAME=$(basename "${PROMPT_FILE}" .txt)
         OUTPUT_DIR="Continual/output_img_Monet/discriminator_${F_SCALE}_${PROMPT_NAME}"
         mkdir -p "$OUTPUT_DIR"

         echo "Running with F_SCALE=${F_SCALE}, PROMPT_FILE=${PROMPT_FILE}, NSAMPLES=${NSAMPLES}"
         
         # nohup으로 백그라운드 실행하고 PID를 저장한 후 wait로 해당 작업 종료까지 대기
         nohup python test_nudity.py \
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

    while [ $(jobs -pr | wc -l) -ge $MAX_CONCURRENCY ]; do
            wait -n
        done
    done
done
