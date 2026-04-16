#!/bin/bash
# siml-09 GPU 0: NudeNet eval for ALL flux2klein + SD3 dirs (alongside FLUX.1-dev gen)
# NudeNet is lightweight (~2GB), can coexist with FLUX.1-dev gen on 97GB GPU
export CUDA_VISIBLE_DEVICES=0
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
LOG="/mnt/home3/yhgil99/unlearning/logs/nudenet_siml09.log"

echo "=== NudeNet Eval ALL START $(date) ===" > "$LOG"

# --- FLUX.2-klein ---
KLEIN="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux2klein_experiments"

echo "[KLEIN BASELINE]" >> "$LOG"
for d in "$KLEIN"/baseline/*/; do
  [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
    echo "  NN: $(basename $d)" >> "$LOG"
    $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 >> "$LOG" 2>&1 || true
  }
done

echo "[KLEIN OURS]" >> "$LOG"
for d in "$KLEIN"/ours/*/; do
  [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
    echo "  NN: $(basename $d)" >> "$LOG"
    $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 >> "$LOG" 2>&1 || true
  }
done

echo "[KLEIN SAFREE]" >> "$LOG"
for d in "$KLEIN"/safree/*/; do
  [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
    echo "  NN: $(basename $d)" >> "$LOG"
    $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 >> "$LOG" 2>&1 || true
  }
done

# --- FLUX.1-dev (baseline already generated) ---
FDEV="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux1dev_experiments"

echo "[FLUX1DEV BASELINE]" >> "$LOG"
for d in "$FDEV"/baseline/*/; do
  [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
    echo "  NN: $(basename $d)" >> "$LOG"
    $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 >> "$LOG" 2>&1 || true
  }
done

# --- SD3 ---
SD3="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"

echo "[SD3 ALL]" >> "$LOG"
for method in baseline safegen safree safe_denoiser sgf; do
  for d in "$SD3/$method"/*/; do
    [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
      echo "  NN: $method/$(basename $d)" >> "$LOG"
      $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 >> "$LOG" 2>&1 || true
    }
  done
done

echo "=== NudeNet Eval ALL DONE $(date) ===" >> "$LOG"
