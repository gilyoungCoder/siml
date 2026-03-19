# 설치(필요 시)
export CUDA_VISIBLE_DEVICES=3

# 실행 예시
python gen.py \
  --prompt_file ./prompts/country_nude_body.txt \
  --output_dir ./outputs/negative_prompt \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --height 512 --width 512 \
  --seed 1234 \
  --negative_prompt "nudity"     # 기본값이라 생략 가능. 끄려면 "" 사용
