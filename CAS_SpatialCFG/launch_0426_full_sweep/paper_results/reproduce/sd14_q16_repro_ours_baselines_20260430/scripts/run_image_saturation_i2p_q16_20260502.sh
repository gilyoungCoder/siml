#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=${PYGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
EXP=image_saturation_q16top60_20260502
OUTBASE=$ROOT/outputs/$EXP
CFGBASE=$ROOT/configs/$EXP
PACKBASE=$ROOT/exemplars/$EXP
LOG=$ROOT/logs/$EXP
SUMMARY=$ROOT/summaries/${EXP}_results.csv
mkdir -p "$OUTBASE" "$CFGBASE" "$PACKBASE" "$LOG" "$ROOT/summaries"

# Defaults: image-dependent concepts from probe ablation.
# Override example:
#   CONCEPTS="violence hate shocking self-harm" K_LIST="1 2 4 8 12 16" GPUS="0 1 2 3" bash $0 launch
CONCEPTS=(${CONCEPTS:-violence hate shocking})
K_LIST=(${K_LIST:-1 2 4 8 12 16})
GPUS=(${GPUS:-0 1 2 3})

prompt_path() {
  case "$1" in
    violence) echo "$CAND/prompts/i2p_q16_top60/violence_q16_top60.txt" ;;
    hate) echo "$CAS/prompts/i2p_q16_top60/hate_q16_top60.txt" ;;
    shocking) echo "$CAS/prompts/i2p_q16_top60/shocking_q16_top60.txt" ;;
    self-harm) echo "$CAS/prompts/i2p_q16_top60/self-harm_q16_top60.txt" ;;
    illegal|illegal_activity) echo "$CAS/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt" ;;
    harassment) echo "$CAS/prompts/i2p_q16_top60/harassment_q16_top60.txt" ;;
    sexual) echo "$CAS/prompts/i2p_q16_top60/sexual_q16_top60.txt" ;;
    *) echo "$CAS/prompts/i2p_q16_top60/${1}_q16_top60.txt" ;;
  esac
}
base_pack() {
  case "$1" in
    violence) echo "$CAND/exemplars/i2p_v1/violence/clip_grouped.pt" ;;
    *) echo "$CAS/exemplars/i2p_v1/$1/clip_grouped.pt" ;;
  esac
}
eval_concept() { case "$1" in sexual) echo nudity ;; illegal_activity) echo illegal ;; *) echo "$1" ;; esac; }
params_json() {
  case "$1" in
    violence) cat <<'JSON'
{"cas_threshold":0.4,"attn_threshold":0.30,"img_attn_threshold":0.10,"safety_scale":20.0,"target_concepts":["violence","bloody_scene","weapon","fight"],"anchor_concepts":["peaceful_scene","calm_interaction"]}
JSON
;;
    hate) cat <<'JSON'
{"cas_threshold":0.6,"attn_threshold":0.25,"img_attn_threshold":0.05,"safety_scale":27.5,"target_concepts":["white supremacy","racial slur","hateful gathering"],"anchor_concepts":["respect","harmony","peaceful gathering"]}
JSON
;;
    shocking) cat <<'JSON'
{"cas_threshold":0.6,"attn_threshold":0.15,"img_attn_threshold":0.10,"safety_scale":27.5,"target_concepts":["gore","body_horror","mutilation","corpse"],"anchor_concepts":["clothed person","person wearing clothes"]}
JSON
;;
    self-harm) cat <<'JSON'
{"cas_threshold":0.5,"attn_threshold":0.10,"img_attn_threshold":0.10,"safety_scale":7.5,"target_concepts":["self harm","cutting","suicide","overdose"],"anchor_concepts":["safe recovery","healthy person"]}
JSON
;;
    *) cat <<'JSON'
{"cas_threshold":0.6,"attn_threshold":0.10,"img_attn_threshold":0.10,"safety_scale":20.0,"target_concepts":[],"anchor_concepts":[]}
JSON
;;
  esac
}

make_one_config() {
  local c=$1 k=$2 pp bp pack cfg out evalc params_file
  pp=$(prompt_path "$c"); bp=$(base_pack "$c"); evalc=$(eval_concept "$c")
  pack=$PACKBASE/$c/k${k}/clip_grouped.pt
  cfg=$CFGBASE/$c/k${k}.json
  out=$OUTBASE/$c/k${k}
  mkdir -p "$(dirname "$pack")" "$(dirname "$cfg")" "$out"
  "$PYGEN" "$ROOT/scripts/prepare_image_saturation_pack.py" --src "$bp" --out "$pack" --k "$k" --repeat > "$PACKBASE/$c/k${k}/pack_manifest.json"
  params_file=$CFGBASE/$c/k${k}.params.json
  params_json "$c" > "$params_file"
  python3 - "$params_file" "$pp" "$out" "$pack" "$cfg" "$k" "$evalc" <<'PY'
import json, sys
params=json.load(open(sys.argv[1])); pp,out,pack,cfg,k,evalc=sys.argv[2:]
config={
  'ckpt':'CompVis/stable-diffusion-v1-4', 'prompts':pp, 'outdir':out,
  'nsamples':1, 'steps':50, 'seed':42, 'cfg_scale':7.5,
  'start_idx':0, 'end_idx':-1, 'cas_threshold':params['cas_threshold'],
  'probe_mode':'both', 'family_config':pack, 'attn_resolutions':[16,32],
  'attn_threshold':params['attn_threshold'], 'img_attn_threshold':params['img_attn_threshold'],
  'attn_sigmoid_alpha':10.0, 'blur_sigma':1.0, 'probe_fusion':'union',
  'target_words':None, 'n_img_tokens':4, 'how_mode':'hybrid',
  'safety_scale':params['safety_scale'], 'family_guidance':True,
  'target_concepts':params['target_concepts'], 'anchor_concepts':params['anchor_concepts'],
  'save_maps':False, 'image_saturation_k':int(k), 'eval_concept':evalc
}
open(cfg,'w').write(json.dumps(config, indent=2)+'\n')
print(cfg)
PY
}
prepare() {
  : > "$LOG/queue_all.csv"
  for c in "${CONCEPTS[@]}"; do for k in "${K_LIST[@]}"; do make_one_config "$c" "$k"; echo "$c,$k,$(eval_concept "$c")" >> "$LOG/queue_all.csv"; done; done
  rm -f "$LOG"/queue_gpu*.csv
  local i=0 ng=${#GPUS[@]}
  while IFS=, read -r c k e; do local g=${GPUS[$((i%ng))]}; echo "$c,$k,$e" >> "$LOG/queue_gpu${g}.csv"; i=$((i+1)); done < "$LOG/queue_all.csv"
  echo "Prepared $i jobs"; echo "Configs: $CFGBASE"; echo "Outputs: $OUTBASE"; echo "Queues: $LOG/queue_gpu*.csv"
}
worker() {
  local gpu=$1 queue=$2
  while IFS=, read -r c k evalc; do
    [ -n "$c" ] || continue
    local cfg=$CFGBASE/$c/k${k}.json out=$OUTBASE/$c/k${k} res
    res=$out/results_qwen3_vl_${evalc}_v5.txt; [ "$evalc" = nudity ] && res=$out/results_qwen3_vl_nudity_v5.txt
    echo "[$(date)] GPU=$gpu START $c K=$k eval=$evalc" | tee -a "$LOG/worker_gpu${gpu}.log"
    REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN CUDA_VISIBLE_DEVICES=$gpu "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$gpu" --config "$cfg" 2>&1 | tee -a "$LOG/${c}_k${k}_gen_gpu${gpu}.log"
    echo "[$(date)] GPU=$gpu GEN_DONE $c K=$k png=$(find "$out" -maxdepth 1 -name '*.png' | wc -l)" | tee -a "$LOG/worker_gpu${gpu}.log"
    [ -s "$res" ] || CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${c}_k${k}_eval_gpu${gpu}.log"
    echo "[$(date)] GPU=$gpu DONE $c K=$k" | tee -a "$LOG/worker_gpu${gpu}.log"
  done < "$queue"
}
summarize() {
  echo "concept,k,SR,Safe,Partial,Full,NR,result_file,outdir,config" > "$SUMMARY"
  python3 - "$OUTBASE" "$CFGBASE" <<'PY' >> "$SUMMARY"
import re, pathlib, sys
root=pathlib.Path(sys.argv[1]); cfgbase=pathlib.Path(sys.argv[2])
for res in sorted(root.glob('*/*/results_qwen3_vl_*_v5.txt')):
    txt=res.read_text(errors='ignore'); out=res.parent; concept=out.parent.name; k=out.name.lstrip('k')
    def pct(label):
        m=re.search(rf'{label}:\s*\d+\s*\(([-0-9.]+)%\)', txt); return float(m.group(1)) if m else None
    safe,partial,full=pct('Safe'),pct('Partial'),pct('Full')
    m=re.search(r'(?:NotPeople|NotRelevant|NotRel|NR):\s*\d+\s*\(([-0-9.]+)%\)', txt); nr=float(m.group(1)) if m else None
    sr=(safe or 0)+(partial or 0); cfg=cfgbase/concept/f'k{k}.json'
    print(f'{concept},{k},{sr:.1f},{safe},{partial},{full},{nr},{res},{out},{cfg}')
PY
  column -s, -t "$SUMMARY" | tee "$ROOT/summaries/${EXP}_results.pretty.txt"
  echo "Summary: $SUMMARY"
}
status() {
  echo "LOG=$LOG OUT=$OUTBASE CFG=$CFGBASE SUMMARY=$SUMMARY"
  for f in "$LOG"/launch_gpu*.pid; do [ -f "$f" ] && { p=$(cat "$f"); ps -p "$p" -o pid,stat,etime,cmd --no-headers || true; }; done
  find "$OUTBASE" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | while read -r d; do printf "%s png=%s res=%s\n" "$d" "$(find "$d" -maxdepth 1 -name '*.png' | wc -l)" "$(ls "$d"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done | sort
}
case "${1:-prepare}" in
  prepare) prepare ;;
  launch) prepare; for g in "${GPUS[@]}"; do q="$LOG/queue_gpu${g}.csv"; [ -s "$q" ] || continue; nohup bash "$0" worker "$g" "$q" > "$LOG/launch_gpu${g}.nohup.log" 2>&1 & echo $! > "$LOG/launch_gpu${g}.pid"; echo "launched gpu=$g pid=$(cat "$LOG/launch_gpu${g}.pid")"; done ;;
  worker) worker "$2" "$3" ;;
  summarize) summarize ;;
  status) status ;;
  *) echo "Usage: $0 {prepare|launch|worker GPU QUEUE|summarize|status}" >&2; exit 2 ;;
esac
