#!/usr/bin/env bash
set -euo pipefail
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
V27=$REPO/CAS_SpatialCFG/generate_v27.py
CLIP=$REPO/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
I2P=$REPO/SAFREE/datasets/i2p_categories
BOTH="--probe_mode both --clip_embeddings $CLIP --attn_threshold 0.1 --img_attn_threshold 0.4"

for concept in violence harassment shocking hate; do
  case $concept in
    violence)   PF=$I2P/i2p_violence.csv; EC=violence
      TW="violence gore bloodshed fighting weapon"; TC="violence gore fighting"; AC="peaceful_scene nature_landscape" ;;
    harassment) PF=$I2P/i2p_harassment.csv; EC=harassment
      TW="harassment bullying threatening intimidation"; TC="harassment bullying intimidation"; AC="friendly_conversation cooperation" ;;
    shocking)   PF=$I2P/i2p_shocking.csv; EC=shocking
      TW="grotesque body_horror disturbing"; TC="grotesque disturbing_imagery"; AC="serene_scene beautiful_art" ;;
    hate)       PF=$I2P/i2p_hate.csv; EC=hate
      TW="hate_speech discrimination racist"; TC="hate_speech discrimination"; AC="diversity harmony equality" ;;
  esac

  ODIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_${concept}_both_hyb_ts15_as15_cas05
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 7: v27 hybrid $concept"
    CUDA_VISIBLE_DEVICES=7 $P $V27 \
      --prompts "$PF" --outdir "$ODIR" \
      --nsamples 1 --steps 50 --seed 42 --cas_threshold 0.5 \
      --how_mode hybrid --target_scale 15 --anchor_scale 15 \
      --target_words $TW --target_concepts $TC --anchor_concepts $AC \
      $BOTH
  fi
  [ ! -f "$ODIR/categories_qwen3_vl_${EC}.json" ] && \
    CUDA_VISIBLE_DEVICES=7 $VLP $VLD/opensource_vlm_i2p_all.py "$ODIR" "$EC" qwen 2>&1 | tail -1
done
echo "GPU7 Hybrid DONE — $(date)"
