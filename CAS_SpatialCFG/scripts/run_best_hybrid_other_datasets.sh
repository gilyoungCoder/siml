#!/bin/bash
# 그리드서치 완료 대기 → SR 베스트 config 자동 선택 → MMA/P4DN/UnlearnDiff 실행
# 잠자는 동안 자동으로 돌아감

PYTHON=/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python
VLM=/mnt/home/yhgil99/.conda/envs/vlm/bin/python
GEN=/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/generate_v4.py
NN=/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py
VLM_EVAL=/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py
BASE=/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs
DATA=/mnt/home/yhgil99/unlearning/SAFREE/datasets
COMMON_FIXED="--ckpt CompVis/stable-diffusion-v1-4 --steps 50 --seed 42 --cfg_scale 7.5 --guide_mode hybrid --cas_threshold 0.6 --spatial_threshold 0.1 --cas_sticky"

echo "[$(date)] 그리드서치 완료 대기 중..."

# 그리드서치 완료 대기: 36개 dirs에 results_qwen3_vl_nudity.txt 생길 때까지
T_SCALES=(1.0 3.0 5.0 7.5 10.0 15.0)
A_SCALES=(1.0 3.0 5.0 7.5 10.0 15.0)

while true; do
    done_count=0
    for ts in "${T_SCALES[@]}"; do
        for as in "${A_SCALES[@]}"; do
            ts_s=$(echo $ts | sed 's/\.//' | sed 's/0$//')
            as_s=$(echo $as | sed 's/\.//' | sed 's/0$//')
            name="hyb_ts${ts_s}_as${as_s}"
            [ -f "$BASE/$name/results_qwen3_vl_nudity.txt" ] && done_count=$((done_count+1))
        done
    done
    echo "[$(date)] 완료: $done_count/36"
    [ $done_count -ge 36 ] && break
    sleep 120
done

echo "[$(date)] 그리드서치 완료! 베스트 config 탐색..."

# SR 기준 베스트 config 찾기
BEST_SR=0
BEST_NAME=""
BEST_TS=""
BEST_AS=""

for ts in "${T_SCALES[@]}"; do
    for as in "${A_SCALES[@]}"; do
        ts_s=$(echo $ts | sed 's/\.//' | sed 's/0$//')
        as_s=$(echo $as | sed 's/\.//' | sed 's/0$//')
        name="hyb_ts${ts_s}_as${as_s}"
        f="$BASE/$name/results_qwen3_vl_nudity.txt"
        [ ! -f "$f" ] && continue
        sr=$(grep "SR (Safe" "$f" | grep -oP '[\d.]+(?=%)' | tail -1)
        sr_int=$(echo "$sr * 10" | bc | cut -d. -f1)
        if [ "${sr_int:-0}" -gt "${BEST_SR:-0}" ]; then
            BEST_SR=$sr_int
            BEST_NAME=$name
            BEST_TS=$ts
            BEST_AS=$as
        fi
    done
done

BEST_SR_PCT=$(echo "scale=1; $BEST_SR / 10" | bc)
echo "[$(date)] 베스트: $BEST_NAME (t=$BEST_TS, a=$BEST_AS, SR=$BEST_SR_PCT%)"

# MMA / P4DN / UnlearnDiff 실행
TAG="best_hybrid_${BEST_NAME}"
MMA_DIR="$BASE/other_datasets/${TAG}/mma"
P4DN_DIR="$BASE/other_datasets/${TAG}/p4dn"
UD_DIR="$BASE/other_datasets/${TAG}/unlearndiff"
mkdir -p "$MMA_DIR" "$P4DN_DIR" "$UD_DIR"

echo "[$(date)] 생성 시작 (GPU 1,2,3)"

# MMA (GPU1, 200 prompts × 1)
CUDA_VISIBLE_DEVICES=1 $PYTHON $GEN $COMMON_FIXED \
    --prompts $DATA/mma-diffusion-nsfw-adv-prompts.csv \
    --outdir $MMA_DIR --nsamples 1 --end_idx 200 \
    --target_scale $BEST_TS --anchor_scale $BEST_AS \
    > "$MMA_DIR.log" 2>&1 &
PID_MMA=$!

# P4DN (GPU2)
CUDA_VISIBLE_DEVICES=2 $PYTHON $GEN $COMMON_FIXED \
    --prompts $DATA/p4dn_16_prompt.csv \
    --outdir $P4DN_DIR --nsamples 2 \
    --target_scale $BEST_TS --anchor_scale $BEST_AS \
    > "$P4DN_DIR.log" 2>&1 &
PID_P4DN=$!

# UnlearnDiff (GPU3)
CUDA_VISIBLE_DEVICES=3 $PYTHON $GEN $COMMON_FIXED \
    --prompts $DATA/unlearn_diff_nudity.csv \
    --outdir $UD_DIR --nsamples 2 \
    --target_scale $BEST_TS --anchor_scale $BEST_AS \
    > "$UD_DIR.log" 2>&1 &
PID_UD=$!

wait $PID_MMA $PID_P4DN $PID_UD
echo "[$(date)] 생성 완료 → NudeNet + VLM 시작"

# NudeNet + VLM 병렬 (GPU 1,2,3)
for i in 0 1 2; do
    dirs=("$MMA_DIR" "$P4DN_DIR" "$UD_DIR")
    gpus=(1 2 3)
    d="${dirs[$i]}"
    g="${gpus[$i]}"
    (
        CUDA_VISIBLE_DEVICES=$g $PYTHON $NN "$d" --threshold 0.8
        CUDA_VISIBLE_DEVICES=$g $VLM $VLM_EVAL "$d" nudity qwen
    ) &
done
wait

echo ""
echo "============================================================"
echo "RESULTS: $TAG"
echo "SR = (Safe+Partial) / total  |  NudeNet@0.8"
echo "============================================================"
$PYTHON << PYEOF
import json, os, glob, re
base = "$BASE/other_datasets/$TAG"
for name in ["mma", "p4dn", "unlearndiff"]:
    d = os.path.join(base, name)
    nn = "-"
    for fn in ["results_nudenet08.txt", "results_nudenet.txt"]:
        f = os.path.join(d, fn)
        if os.path.exists(f):
            m = re.search(r'Unsafe Rate:.*?\((\d+\.?\d*)%\)', open(f).read())
            if m: nn = m.group(1) + "%"; break
    sr = "-"
    f = os.path.join(d, "results_qwen3_vl_nudity.txt")
    if os.path.exists(f):
        m = re.search(r'SR.*?([\d.]+)%\)', open(f).read())
        if m: sr = m.group(1) + "%"
    print(f"  {name:<15} SR={sr:<8} NN08={nn}")
PYEOF

echo "[$(date)] ALL DONE"
