#!/bin/bash

# GenEval 평가 예시 스크립트
# 실제 경로로 수정하여 사용하세요

# 예시 1: 기본 사용 (CUDA)
# ./run_geneval.sh \
#     --img_dir ./generated_images \
#     --prompt_file ./prompts.txt \
#     --output ./results.json

# 예시 2: CPU 사용
# ./run_geneval.sh \
#     --img_dir ./generated_images \
#     --prompt_file ./prompts.json \
#     --output ./results_cpu.json \
#     --device cpu

# 예시 3: 절대 경로 사용
# ./run_geneval.sh \
#     --img_dir /mnt/home/yhgil99/unlearning/SoftDelete+CG/generated_images \
#     --prompt_file /mnt/home/yhgil99/unlearning/SoftDelete+CG/prompts.txt \
#     --output /mnt/home/yhgil99/unlearning/SoftDelete+CG/eval_results.json

# 예시 4: 여러 실험 배치 평가
# for exp_dir in experiment_*/; do
#     exp_name=$(basename "$exp_dir")
#     echo "Evaluating $exp_name..."
#     ./run_geneval.sh \
#         --img_dir "${exp_dir}/images" \
#         --prompt_file "${exp_dir}/prompts.txt" \
#         --output "${exp_dir}/geneval_results.json"
# done

echo "이 파일은 예시 스크립트입니다."
echo "위의 주석을 참고하여 실제 경로로 수정한 후 사용하세요."
echo ""
echo "사용법:"
echo "./run_geneval.sh --img_dir <이미지_디렉토리> --prompt_file <프롬프트_파일>"
