#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
CFGDIR=$ROOT/configs/probe_ablation_q16top60_20260501_both_violence_adaptive
OUTROOT=$ROOT/outputs/probe_ablation_q16top60_20260501_both_violence_adaptive
LOGDIR=$ROOT/logs/probe_ablation_q16top60_20260501_both_violence_adaptive
PYGEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
SUMMARY=$ROOT/summaries/probe_ablation_violence_both_adaptive_20260501.md
mkdir -p "$CFGDIR" "$OUTROOT" "$LOGDIR" "$ROOT/summaries"
echo "# Adaptive both/hybrid violence sweep (I2P q16 top-60)" > "$SUMMARY"
echo >> "$SUMMARY"
echo "| Variant | sh | tau | txt | img | SR | Safe | Partial | Full | NR | path |" >> "$SUMMARY"
echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|" >> "$SUMMARY"
while pgrep -af "sh20_tau04_txt030_img010.*run_from_config.py|sh20_tau04_txt030_img010.*safegen.generate_family|run_sh20_txt030_img010_siml05_4567" >/dev/null; do
  echo "[$(date)] waiting for existing sh20 4-way shard to finish" | tee -a "$LOGDIR/worker_adaptive_4567.log"
  sleep 60
done
run_candidate(){
  local name=$1 sh=$2 tau=$3 txt=$4 img=$5
  local out=$OUTROOT/$name
  mkdir -p "$out"
  rm -f "$out/results_qwen3_vl_violence_v5.txt" "$out/categories_qwen3_vl_violence_v5.json"
  for spec in "4 0 15" "5 15 30" "6 30 45" "7 45 60"; do
    set -- $spec; local gpu=$1 start=$2 end=$3
    local cfg=$CFGDIR/${name}_${start}_${end}.json
    cat > "$cfg" <<JSON
{
  "ckpt": "CompVis/stable-diffusion-v1-4",
  "prompts": "$CAND/prompts/i2p_q16_top60/violence_q16_top60.txt",
  "outdir": "$out",
  "nsamples": 1,
  "steps": 50,
  "seed": 42,
  "cfg_scale": 7.5,
  "start_idx": $start,
  "end_idx": $end,
  "cas_threshold": $tau,
  "probe_mode": "both",
  "family_config": "$CAND/exemplars/i2p_v1/violence/clip_grouped.pt",
  "attn_threshold": $txt,
  "img_attn_threshold": $img,
  "attn_sigmoid_alpha": 10.0,
  "blur_sigma": 1.0,
  "probe_fusion": "union",
  "target_words": ["violence", "bloody", "scene", "weapon", "fight"],
  "n_img_tokens": 4,
  "how_mode": "hybrid",
  "safety_scale": $sh,
  "family_guidance": true,
  "target_concepts": ["violence", "bloody_scene", "weapon", "fight"],
  "anchor_concepts": ["peaceful_scene", "calm_interaction"],
  "attn_resolutions": [16, 32],
  "save_maps": false,
  "note": "adaptive violence both/hybrid sweep, full-vs-NR guided"
}
JSON
    echo "[$(date)] START $name shard gpu=$gpu $start:$end sh=$sh tau=$tau txt=$txt img=$img" | tee -a "$LOGDIR/worker_adaptive_4567.log"
    CUDA_VISIBLE_DEVICES=$gpu REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN \
      "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$gpu" --config "$cfg" \
      > "$LOGDIR/${name}_gpu${gpu}_${start}_${end}_gen.log" 2>&1 &
  done
  wait
  local n=$(find "$out" -maxdepth 1 -name "*.png" | wc -l)
  echo "[$(date)] GEN_DONE $name n=$n" | tee -a "$LOGDIR/worker_adaptive_4567.log"
  CUDA_VISIBLE_DEVICES=4 "$PYVLM" "$V5" "$out" violence qwen > "$LOGDIR/${name}_eval.log" 2>&1
  cat "$out/results_qwen3_vl_violence_v5.txt" | tee -a "$LOGDIR/worker_adaptive_4567.log"
  python3 - "$name" "$sh" "$tau" "$txt" "$img" "$out" "$SUMMARY" <<'PY'
import sys,re,pathlib
name,sh,tau,txt,img,out,summary=sys.argv[1:]
t=(pathlib.Path(out)/'results_qwen3_vl_violence_v5.txt').read_text()
def get(label):
    m=re.search(rf"- {label}: \d+ \(([0-9.]+)%\)",t); return float(m.group(1)) if m else -1
m=re.search(r"SR \(Safe\+Partial\): \d+/\d+ \(([0-9.]+)%\)",t); sr=float(m.group(1)) if m else -1
safe,partial,full,nr=get('Safe'),get('Partial'),get('Full'),get('NotRelevant')
with open(summary,'a') as f:
    f.write(f"| {name} | {sh} | {tau} | {txt} | {img} | {sr:.1f} | {safe:.1f} | {partial:.1f} | {full:.1f} | {nr:.1f} | `{out}` |\n")
print(f"PARSED {name}: SR={sr} Full={full} NR={nr}")
PY
}
run_candidate sh23_tau04_txt015_img010 23 0.4 0.15 0.10
run_candidate sh24_tau04_txt015_img010 24 0.4 0.15 0.10
run_candidate sh23_tau04_txt020_img010 23 0.4 0.20 0.10
run_candidate sh24_tau04_txt020_img010 24 0.4 0.20 0.10
run_candidate sh22_tau04_txt030_img010 22 0.4 0.30 0.10
run_candidate sh23_tau04_txt030_img010 23 0.4 0.30 0.10
run_candidate sh24_tau04_txt030_img010 24 0.4 0.30 0.10
run_candidate sh23_tau04_txt030_img015 23 0.4 0.30 0.15
run_candidate sh24_tau04_txt030_img015 24 0.4 0.30 0.15
run_candidate sh25_tau04_txt020_img015 25 0.4 0.20 0.15
run_candidate sh25_tau038_txt020_img010 25 0.38 0.20 0.10
run_candidate sh25_tau042_txt020_img010 25 0.42 0.20 0.10
python3 - "$SUMMARY" <<'PY' | tee -a "$LOGDIR/worker_adaptive_4567.log"
import sys,pathlib
p=pathlib.Path(sys.argv[1]); rows=[]
for line in p.read_text().splitlines():
    if line.startswith('| sh'):
        parts=[x.strip() for x in line.strip('|').split('|')]
        rows.append(parts)
rows.sort(key=lambda r:(-float(r[5]), float(r[9]), float(r[8])))
print('\nBEST_SORTED')
for r in rows[:12]: print(r)
PY
