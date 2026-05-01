#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY_OFFICIAL=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
PY_OURS=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
TXT=$ROOT/prompts/runtime/runtime_sexual_10.txt
CSV=$ROOT/prompts/runtime/runtime_sexual_10.csv
OUTROOT=$ROOT/outputs/nfe/sexual10_5methods_20260502
LOGDIR=$ROOT/logs/nfe/sexual10_5methods_20260502
SUMMARY=$ROOT/summaries/nfe_sexual10_5methods_20260502.md
mkdir -p "$OUTROOT" "$LOGDIR" "$ROOT/summaries"
STEPS="10 20 30 40 50"
run_baseline(){
  local gpu=0
  for s in $STEPS; do
    local out=$OUTROOT/baseline/nfe${s}
    rm -rf "$out"; mkdir -p "$out"
    echo "[$(date)] baseline nfe=$s gpu=$gpu" | tee -a "$LOGDIR/baseline.log"
    CUDA_VISIBLE_DEVICES=$gpu "$PY_OFFICIAL" "$ROOT/scripts/runtime_baseline_sd14.py" --prompts "$TXT" --outdir "$out" --steps "$s" --cfg 7.5 --seed 42 > "$LOGDIR/baseline_nfe${s}.log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" nudity qwen > "$LOGDIR/baseline_nfe${s}_eval.log" 2>&1
  done
}
run_official(){
  local method=$1 gpu=$2 script=$3 repo=$4 task=$5 erase=$6
  for s in $STEPS; do
    local out=$OUTROOT/$method/nfe${s}
    rm -rf "$out"; mkdir -p "$out/safe" "$out/unsafe" "$out/all"
    echo "[$(date)] $method nfe=$s gpu=$gpu" | tee -a "$LOGDIR/${method}.log"
    (cd "$repo" && CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 "$PY_OFFICIAL" "$script" \
      --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps="$s" \
      --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
      --data="$CSV" --category=all --task_config="$task" --save-dir="$out" \
      --erase_id="$erase" --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,10 \
      > "$LOGDIR/${method}_nfe${s}.log" 2>&1)
    CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out/all" nudity qwen > "$LOGDIR/${method}_nfe${s}_eval.log" 2>&1
  done
}
run_ours(){
  local gpu=5
  for s in $STEPS; do
    local out=$OUTROOT/ours/nfe${s}
    rm -rf "$out"; mkdir -p "$out"
    echo "[$(date)] ours nfe=$s gpu=$gpu" | tee -a "$LOGDIR/ours.log"
    (cd "$ROOT/code/SafeGen" && CUDA_VISIBLE_DEVICES=$gpu "$PY_OURS" -m safegen.generate_family \
      --ckpt CompVis/stable-diffusion-v1-4 --prompts "$TXT" --outdir "$out" \
      --nsamples 1 --steps "$s" --seed 42 --cfg_scale 7.5 --start_idx 0 --end_idx 10 \
      --cas_threshold 0.5 --probe_mode both --family_config /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt \
      --attn_threshold 0.1 --img_attn_threshold 0.3 --attn_sigmoid_alpha 10.0 --blur_sigma 1.0 \
      --probe_fusion union --n_img_tokens 4 --how_mode hybrid --safety_scale 20.0 --attn_resolutions 16 32 \
      --target_words nudity nude person naked body --family_guidance --target_concepts nudity "nude person" "naked body" --anchor_concepts "clothed person" "person wearing clothes" \
      > "$LOGDIR/ours_nfe${s}.log" 2>&1)
    CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" nudity qwen > "$LOGDIR/ours_nfe${s}_eval.log" 2>&1
  done
}
summarize(){
  python3 - <<'PY'
import re, json
from pathlib import Path
root=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
outroot=root/'outputs/nfe/sexual10_5methods_20260502'
summary=root/'summaries/nfe_sexual10_5methods_20260502.md'
methods=['baseline','safree','safedenoiser','sgf','ours']
steps=[10,20,30,40,50]
def parse_eval(p):
    if not p.exists(): return None
    t=p.read_text(errors='ignore')
    def pct(label):
        m=re.search(rf'- {label}: \d+ \(([0-9.]+)%\)', t); return float(m.group(1)) if m else None
    m=re.search(r'SR \(Safe\+Partial\): \d+/\d+ \(([0-9.]+)%\)', t)
    return {'SR':float(m.group(1)) if m else None,'Safe':pct('Safe'),'Partial':pct('Partial'),'Full':pct('Full'),'NR':pct('NotPeople')}
lines=['# NFE sweep sexual10 5 methods (2026-05-02)','', '| Method | NFE | SR | Safe | Partial | Full | NR | Result path |','|---|---:|---:|---:|---:|---:|---:|---|']
for method in methods:
    for s in steps:
        d=outroot/method/f'nfe{s}'
        eval_file=d/'results_qwen3_vl_nudity_v5.txt'
        if method in ['safree','safedenoiser','sgf']:
            eval_file=d/'all/results_qwen3_vl_nudity_v5.txt'
        r=parse_eval(eval_file)
        if r:
            lines.append(f"| {method} | {s} | {r['SR']:.1f} | {r['Safe']:.1f} | {r['Partial']:.1f} | {r['Full']:.1f} | {r['NR']:.1f} | `{eval_file}` |")
        else:
            lines.append(f"| {method} | {s} | TBD | TBD | TBD | TBD | TBD | `{eval_file}` |")
summary.write_text('\n'.join(lines)+'\n')
print(summary)
PY
}
case "${1:-all}" in
 baseline) run_baseline;;
 safree) run_official safree 2 run_copro.py "$ROOT/code/official_repos/Safe_Denoiser" configs/nudity/safe_denoiser.yaml safree_neg_prompt;;
 safedenoiser) run_official safedenoiser 3 run_copro.py "$ROOT/code/official_repos/Safe_Denoiser" configs/nudity/safe_denoiser.yaml safree_neg_prompt_rep_threshold_time;;
 sgf) run_official sgf 4 generate_unsafe_sgf.py "$ROOT/code/official_repos/SGF/nudity_sdv1" configs/sgf/sgf.yaml safree_neg_prompt_rep_time;;
 ours) run_ours;;
 summarize) summarize;;
 all) run_baseline & run_official safree 2 run_copro.py "$ROOT/code/official_repos/Safe_Denoiser" configs/nudity/safe_denoiser.yaml safree_neg_prompt & run_official safedenoiser 3 run_copro.py "$ROOT/code/official_repos/Safe_Denoiser" configs/nudity/safe_denoiser.yaml safree_neg_prompt_rep_threshold_time & run_official sgf 4 generate_unsafe_sgf.py "$ROOT/code/official_repos/SGF/nudity_sdv1" configs/sgf/sgf.yaml safree_neg_prompt_rep_time & run_ours & wait; summarize;;
 *) echo bad mode; exit 2;;
esac
