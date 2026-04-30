#!/usr/bin/env bash
set -uo pipefail
GPU=${1:-0}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY_OFFICIAL=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
PY_OURS=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
CSV=$ROOT/prompts/runtime/runtime_sexual_10.csv
TXT=$ROOT/prompts/runtime/runtime_sexual_10.txt
LOGDIR=$ROOT/logs/runtime
OUTROOT=$ROOT/outputs/runtime/sexual10
mkdir -p "$LOGDIR" "$OUTROOT" "$ROOT/summaries"
run_official_safe() {
  local name=$1 erase=$2 task=$3
  local out=$OUTROOT/$name
  rm -rf "$out"; mkdir -p "$out/safe" "$out/unsafe" "$out/all"
  cd "$ROOT/code/official_repos/Safe_Denoiser"
  echo "RUN_RUNTIME $name"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY_OFFICIAL" run_copro.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
    --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
    --data="$CSV" --category=all --task_config="$task" --save-dir="$out" \
    --erase_id="$erase" --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,10 \
    2>&1 | tee "$LOGDIR/${name}.log"
}
run_sgf() {
  local out=$OUTROOT/sgf
  rm -rf "$out"; mkdir -p "$out/safe" "$out/unsafe" "$out/all"
  cd "$ROOT/code/official_repos/SGF/nudity_sdv1"
  echo "RUN_RUNTIME sgf"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY_OFFICIAL" generate_unsafe_sgf.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
    --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
    --data="$CSV" --category=all --task_config=configs/sgf/sgf.yaml --save-dir="$out" \
    --erase_id=safree_neg_prompt_rep_time --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,10 \
    2>&1 | tee "$LOGDIR/sgf.log"
}
run_baseline() {
  echo "RUN_RUNTIME baseline_sd14"
  CUDA_VISIBLE_DEVICES=$GPU "$PY_OFFICIAL" "$ROOT/scripts/runtime_baseline_sd14.py" --prompts "$TXT" --outdir "$OUTROOT/baseline_sd14" --steps 50 --cfg 7.5 --seed 42 2>&1 | tee "$LOGDIR/baseline_sd14.log"
}
run_ours() {
  local cfg=$ROOT/configs/runtime_ours_sexual10.json
  python3 - <<PY
import json
from pathlib import Path
root=Path('$ROOT')
base=json.loads((root/'configs/ours/i2p_q16/sexual.json').read_text())
base['prompts']=str(root/'prompts/runtime/runtime_sexual_10.txt')
base['outdir']=str(root/'outputs/runtime/sexual10/ours_hybrid')
base['end_idx']=10
(root/'configs/runtime_ours_sexual10.json').write_text(json.dumps(base, indent=2))
PY
  rm -rf "$OUTROOT/ours_hybrid"
  cd "$ROOT/code/SafeGen"
  echo "RUN_RUNTIME ours_hybrid"
  CUDA_VISIBLE_DEVICES=$GPU "$PY_OURS" -m safegen.generate_family_timed \
    --ckpt CompVis/stable-diffusion-v1-4 --prompts "$TXT" --outdir "$OUTROOT/ours_hybrid" \
    --nsamples 1 --steps 50 --seed 42 --cfg_scale 7.5 --start_idx 0 --end_idx 10 \
    --cas_threshold 0.6 --probe_mode both --family_config /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt \
    --attn_threshold 0.1 --img_attn_threshold 0.3 --attn_sigmoid_alpha 10.0 --blur_sigma 1.0 \
    --probe_fusion union --n_img_tokens 4 --how_mode hybrid --safety_scale 20.0 --attn_resolutions 16 32 \
    --target_words nudity nude person naked body --family_guidance --target_concepts nudity "nude person" "naked body" --anchor_concepts "clothed person" "person wearing clothes" \
    2>&1 | tee "$LOGDIR/ours_hybrid.log"
}
summarize() {
  "$PY_OFFICIAL" - <<'PY'
import re, json
from pathlib import Path
root=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
rows=[]
for name in ['baseline_sd14','safree','safedenoiser','sgf','ours_hybrid']:
    log=root/'logs/runtime'/f'{name}.log'
    txt=log.read_text(errors='ignore') if log.exists() else ''
    times=[float(x) for x in re.findall(r'Wall-Clock Time for image generation \(Case#: .*?\): ([0-9.]+) seconds', txt)]
    rows.append((name,len(times),sum(times)/len(times) if times else None,times))
out=root/'summaries/runtime_sd14_sexual10_5methods.md'
with out.open('w') as f:
    f.write('# Runtime SD1.4 sexual10 quick benchmark\n\n| method | n | mean sec/img |\n|---|---:|---:|\n')
    for name,n,mean,times in rows:
        f.write(f'| {name} | {n} | {mean:.4f} |\n' if mean is not None else f'| {name} | {n} | NA |\n')
    f.write('\n```json\n')
    f.write(json.dumps({name:{'n':n,'mean':mean,'times':times} for name,n,mean,times in rows}, indent=2))
    f.write('\n```\n')
print(out)
PY
}
run_baseline
run_official_safe safree safree_neg_prompt configs/nudity/safe_denoiser.yaml
run_official_safe safedenoiser safree_neg_prompt_rep_threshold_time configs/nudity/safe_denoiser.yaml
run_sgf
run_ours
summarize
