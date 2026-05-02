#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 설정 (필요시 변경)
export CUDA_VISIBLE_DEVICES=6

# Conda/venv 환경 활성화 예시
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate sdd

# 실행
python steer_sd14.py

echo "실험 완료! 결과는 outputs/ 디렉토리에 저장됩니다."
echo " - original.png : 기본 Stable Diffusion v1.4 결과"
echo " - steered.png  : Steering 적용 결과"
echo " - steering_strength.png : 토큰별 harm/safe energy 및 γ,λ 시각화"
