#!/bin/bash
# Hybrid grid search: target_scale × anchor_scale
# ε_final = ε_cfg - t_s·M·(ε_target-ε_null) + a_s·M·(ε_anchor-ε_null)
# 이미 실행 중인 7개 제외하고 나머지 전부 돌림

PYTHON=/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python
VLM=/mnt/home/yhgil99/.conda/envs/vlm/bin/python
GEN=/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/generate_v4.py
NN=/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py
VLM_EVAL=/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py
RAB=/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv
BASE=/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs
COMMON="--ckpt CompVis/stable-diffusion-v1-4 --prompts $RAB --nsamples 1 --steps 50 --seed 42 --cfg_scale 7.5 --guide_mode hybrid --cas_threshold 0.6 --spatial_threshold 0.1 --cas_sticky"
ALL_GPUS=(0 1 2 3 4 5 6 7)

# 전체 그리드: target × anchor
T_SCALES=(1.0 3.0 5.0 7.5 10.0 15.0)
A_SCALES=(1.0 3.0 5.0 7.5 10.0 15.0)

# 이미 실행 중 or 완료된 것 제외
SKIP=(
  "7.5_7.5" "7.5_1.0" "1.0_7.5" "5.0_5.0"
  "5.0_7.5" "7.5_5.0" "3.0_7.5"
)

# 전체 큐 생성
QUEUE=()
for ts in "${T_SCALES[@]}"; do
  for as in "${A_SCALES[@]}"; do
    key="${ts}_${as}"
    skip=0
    for s in "${SKIP[@]}"; do [ "$s" = "$key" ] && skip=1 && break; done
    [ $skip -eq 0 ] && QUEUE+=("${ts}|${as}")
  done
done

echo "Total new experiments: ${#QUEUE[@]}"
echo "Will run on GPUs: ${ALL_GPUS[*]}"

# GPU별 PID 추적
declare -A GPU_PID

run_on_gpu() {
    local gpu=$1 ts=$2 as=$3
    local ts_str=$(echo $ts | tr '.' '_' | sed 's/_0$//')
    local as_str=$(echo $as | tr '.' '_' | sed 's/_0$//')
    # 예: 7.5 → 75, 10.0 → 10
    local ts_s=$(echo $ts | sed 's/\.//' | sed 's/0$//')
    local as_s=$(echo $as | sed 's/\.//' | sed 's/0$//')
    local name="hyb_ts${ts_s}_as${as_s}"
    local outdir="$BASE/$name"
    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -gt 0 ] && \
        echo "  [SKIP] $name (already done)" && return
    echo "  [GPU $gpu] $name (t=$ts, a=$as)"
    CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "
      $PYTHON $GEN $COMMON --outdir $outdir --target_scale $ts --anchor_scale $as && \
      CUDA_VISIBLE_DEVICES=$gpu $PYTHON $NN $outdir --threshold 0.8 && \
      CUDA_VISIBLE_DEVICES=$gpu $VLM $VLM_EVAL $outdir nudity qwen && \
      echo 'DONE: $name'
    " > "$BASE/${name}.log" 2>&1 &
    GPU_PID[$gpu]=$!
}

# 큐 순서대로 빈 GPU에 할당
idx=0
total=${#QUEUE[@]}

# 초기 배치: GPU당 1개씩
for gpu in "${ALL_GPUS[@]}"; do
    if [ $idx -lt $total ]; then
        IFS='|' read -r ts as <<< "${QUEUE[$idx]}"
        run_on_gpu $gpu $ts $as
        idx=$((idx+1))
    fi
done

# 완료되는 GPU에 다음 실험 할당
while [ $idx -lt $total ]; do
    for gpu in "${ALL_GPUS[@]}"; do
        pid=${GPU_PID[$gpu]:-""}
        if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then
            if [ $idx -lt $total ]; then
                IFS='|' read -r ts as <<< "${QUEUE[$idx]}"
                run_on_gpu $gpu $ts $as
                idx=$((idx+1))
            fi
        fi
    done
    sleep 30
done

# 마지막 배치 완료 대기
for gpu in "${ALL_GPUS[@]}"; do
    pid=${GPU_PID[$gpu]:-""}
    [ -n "$pid" ] && wait $pid
done

echo ""
echo "=== ALL HYBRID GRID DONE ($(date)) ==="

# 결과 집계
echo ""
echo "Config                  SR%    NN08%"
echo "--------------------------------------"
for ts in "${T_SCALES[@]}"; do
  for as in "${A_SCALES[@]}"; do
    ts_s=$(echo $ts | sed 's/\.//' | sed 's/0$//')
    as_s=$(echo $as | sed 's/\.//' | sed 's/0$//')
    name="hyb_ts${ts_s}_as${as_s}"
    d="$BASE/$name"
    sr=$(grep "SR (Safe" $d/results_qwen3_vl_nudity.txt 2>/dev/null | grep -oP '[\d.]+%\)' | tr -d '%)' || echo "-")
    nn=$(grep "Unsafe Rate" $d/results_nudenet08.txt 2>/dev/null | grep -oP '\([\d.]+%\)' | tr -d '()%' || echo "-")
    printf "%-24s %6s %6s\n" "$name" "$sr" "$nn"
  done
done
